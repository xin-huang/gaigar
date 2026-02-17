# Copyright 2025 Xin Huang
#
# GNU General Public License v3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, please see
#
#    https://www.gnu.org/licenses/gpl-3.0.en.html


import h5py, os, pickle
import shutil, time
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from scipy.special import expit
from sklearn.metrics import accuracy_score
from torch.nn import BCEWithLogitsLoss

from gaishi.models import MlModel
from gaishi.models.unet.layers import UNetPlusPlus, UNetPlusPlusRNN
from gaishi.registries.model_registry import MODEL_REGISTRY

from gaishi.models.unet.dataloader_h5 import build_dataloaders_from_h5


@MODEL_REGISTRY.register("unet")
class UNetModel(MlModel):
    """
    UNet based model wrapper for training and inference on key chunked HDF5 datasets.

    This class provides a minimal public API with static methods. The implementation
    assumes the training and evaluation data are stored in an HDF5 file where each
    top level key corresponds to one chunk of samples and contains at least an input
    dataset ``x_0`` and, for labeled data, a label dataset ``y``.

    Notes
    -----
    - Training uses a key level train validation split and constructs PyTorch
      DataLoaders that build batches by concatenating multiple key chunks.
    - Class imbalance is handled via ``pos_weight`` in ``BCEWithLogitsLoss``.
    - The best model weights are selected by minimum validation loss and written
      to ``{model_dir}/best.weights``.
    - Validation keys are saved to ``{model_dir}/val_keys.pkl`` for reproducibility.
    - Model selection:
        * add_channels == False -> UNetPlusPlus(num_classes=n_classes, input_channels=2)
        * add_channels == True  -> UNetPlusPlusRNN(polymorphisms=W) with 4-channel input
    """

    @staticmethod
    def train(
        data: str,
        output: str,
        trained_model_file: Optional[str] = None,
        add_channels: bool = False,
        n_classes: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        label_noise: float = 0.01,
        n_early: int = 10,
        n_epochs: int = 100,
        min_delta: float = 1e-4,
        label_smooth: bool = True,
        val_prop: float = 0.05,
        seed: int = None,
    ) -> None:
        """
        Train a UNet model on a replicate-indexed HDF5 dataset and save the best weights.

        This training routine assumes the unified (replicate-first) HDF5 schema produced by
        ``write_h5(..., ds_type="train")``. The HDF5 file stores all replicates/windows as
        dense datasets under fixed paths. Each replicate is one training sample.

        HDF5 schema (training)
        ----------------------
        Inputs (always required)
        - ``/data/Ref_genotype`` : uint32, shape (n, N, L)
        - ``/data/Tgt_genotype`` : uint32, shape (n, N, L)
        - ``/data/Gap_to_prev``  : int64,  shape (n, N, L)
        - ``/data/Gap_to_next``  : int64,  shape (n, N, L)

        Targets (required for training)
        - ``/targets/Label``     : uint8,  shape (n, N, L)

        Metadata
        - ``/meta`` attributes include ``n``, ``N`` and ``L``.

        Batch semantics
        --------------
        The DataLoader draws individual replicates and stacks them into batches. With
        ``batch_size=B`` and ``add_channels``:

        - If ``add_channels=False``: model inputs are constructed as 2 channels
          ``[Ref_genotype, Tgt_genotype]`` and the batch tensor has shape ``(B, 2, N, L)``.
        - If ``add_channels=True``: model inputs are constructed as 4 channels
          ``[Ref_genotype, Tgt_genotype, Gap_to_prev, Gap_to_next]`` and the batch tensor
          has shape ``(B, 4, N, L)``.

        Labels are loaded from ``/targets/Label`` and collated as ``(B, 1, N, L)``.
        During training, the label channel dimension is removed via ``y = y.squeeze(1)``
        to match a binary-logit output of shape ``(B, N, L)``.

        Train/validation split
        ----------------------
        Replicates are split deterministically into training and validation subsets by
        shuffling replicate indices with ``seed`` and taking ``val_prop`` as validation.
        The selected indices are returned by ``build_dataloaders_from_h5``.

        Class imbalance
        ---------------
        A negative-to-positive ratio is computed over the training replicates only from
        ``/targets/Label`` and used as ``pos_weight`` in ``BCEWithLogitsLoss``.

        Outputs
        -------
        1. ``output``: model weights with the lowest validation loss
        2. ``training.log``: periodic training loss and accuracy
        3. ``validation.log``: validation loss and accuracy per epoch

        Parameters
        ----------
        data : str
            Path to the HDF5 training file (unified schema).
        output : str
            Path to the best weight file.
        trained_model_file : Optional[str], optional
            If provided, initialize model weights from this file before training.
        add_channels : bool, optional
            If False, use 2-channel inputs (ref, tgt). If True, use 4-channel inputs
            (ref, tgt, gap_to_prev, gap_to_next).
        n_classes : int, optional
            Number of output classes. For binary classification this is typically 1.
            ``UNetPlusPlusRNN`` currently requires ``n_classes == 1``.
        learning_rate : float, optional
            Learning rate for Adam.
        batch_size : int, optional
            Number of replicates per optimization step.
        label_noise : float, optional
            Noise magnitude used for label smoothing during training.
        n_early : int, optional
            Early stopping patience in epochs.
        n_epochs : int, optional
            Maximum number of epochs.
        min_delta : float, optional
            Minimum decrease in validation loss to be considered an improvement.
        label_smooth : bool, optional
            Whether to apply label smoothing to training labels.
        val_prop : float, optional
            Fraction of replicates assigned to validation.
        seed : int, optional
            Seed used for deterministic train/validation split and for label smoothing.

        Raises
        ------
        ValueError
            If the HDF5 file contains no replicates.
        ValueError
            If training labels contain no positive class.
        ValueError
            If ``add_channels`` is True but ``n_classes`` is not 1.
        KeyError
            If required datasets are missing from the HDF5 file.
        """
        start_time = time.time()
        output_dir = os.path.dirname(output)
        os.makedirs(output_dir, exist_ok=True)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        training_log_file = open(os.path.join(output_dir, "training.log"), "w")
        validation_log_file = open(os.path.join(output_dir, "validation.log"), "w")

        # Read shapes from unified schema
        with h5py.File(data, "r") as f:
            n_reps = f["/meta"].attrs["n"]
            N = f["/meta"].attrs["N"]
            L = f["/meta"].attrs["L"]

        if n_reps == 0:
            raise ValueError(f"No replicates found in HDF5 file: {data}")

        input_channels = 4 if add_channels else 2

        train_loader, val_loader, train_indices, val_indices = (
            build_dataloaders_from_h5(
                h5_file=data,
                channels=input_channels,
                batch_size=batch_size,
                val_prop=val_prop,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                seed=seed,
                train_label_smooth=label_smooth,
                train_label_noise=float(label_noise),
            )
        )

        # Compute negative to positive ratio on training indices only
        all_counts0 = 0
        all_counts1 = 0
        with h5py.File(data, "r") as f:
            y_ds = f["/targets/Label"]  # (n, N, L)
            for r in train_indices:
                y = np.asarray(y_ds[int(r)], dtype=np.uint8)
                c1 = int(y.sum())
                all_counts1 += c1
                all_counts0 += int(y.size - c1)

        if all_counts1 == 0:
            raise ValueError(
                "Training labels contain no positive class, all_counts1 is 0."
            )

        ratio = all_counts0 / all_counts1

        if add_channels:
            if int(n_classes) != 1:
                raise ValueError(
                    "UNetPlusPlusRNN currently supports n_classes == 1 only."
                )
            model = UNetPlusPlusRNN(polymorphisms=L)
        else:
            model = UNetPlusPlus(num_classes=int(n_classes), input_channels=2)

        model = model.to(device)

        if trained_model_file is not None:
            checkpoint = torch.load(trained_model_file, map_location=device)
            model.load_state_dict(checkpoint)

        criterion = BCEWithLogitsLoss(pos_weight=torch.FloatTensor([ratio]).to(device))
        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

        min_val_loss = np.inf
        early_count = 0
        best_epoch = 0

        for epoch_idx in range(1, int(n_epochs) + 1):
            model.train()
            losses = []
            accuracies = []

            for batch_idx, (x, y) in enumerate(train_loader, start=1):
                optimizer.zero_grad()

                y = y.squeeze(1)
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                y_pred_bin = np.round(expit(y_pred.detach().cpu().numpy().flatten()))
                y_bin = np.round(y.detach().cpu().numpy().flatten())

                accuracies.append(accuracy_score(y_bin.flatten(), y_pred_bin.flatten()))

                mean_loss = np.mean(losses)
                mean_acc = np.mean(accuracies)

                if batch_idx % 1000 == 0:
                    training_log_file.write(
                        f"Epoch {epoch_idx}, batch {batch_idx}: loss = {mean_loss}, accuracy = {mean_acc}.\n"
                    )
                    training_log_file.flush()

            model.eval()
            val_losses = []
            val_accs = []

            for _, (x, y) in enumerate(val_loader):
                with torch.no_grad():
                    y = torch.squeeze(y)

                    x = x.to(device)
                    y = y.to(device)

                    y_pred = model(x)
                    loss = criterion(y_pred, y)

                    y_pred_bin = np.round(
                        expit(y_pred.detach().cpu().numpy().flatten())
                    )
                    y_bin = np.round(y.detach().cpu().numpy().flatten())

                    val_accs.append(
                        accuracy_score(y_bin.flatten(), y_pred_bin.flatten())
                    )
                    val_losses.append(loss.detach().item())

            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)

            validation_log_file.write(
                f"Epoch {epoch_idx}: validation loss = {val_loss}, validation accuracy = {val_acc}.\n"
            )
            validation_log_file.flush()

            improved = (min_val_loss - val_loss) > float(min_delta)

            if improved:
                min_val_loss = val_loss
                best_epoch = epoch_idx
                validation_log_file.write(
                    f"Best weights saved at epoch {best_epoch}.\n"
                )
                validation_log_file.flush()
                torch.save(model.state_dict(), output)
                early_count = 0
            else:
                early_count += 1
                if early_count >= int(n_early):
                    validation_log_file.write(
                        f"Early stopping; best weights at epoch {best_epoch} reloaded.\n"
                    )
                    validation_log_file.flush()
                    model.load_state_dict(torch.load(output, map_location="cpu"))
                    break

        total = time.time() - start_time
        training_log_file.write(
            f"Training finished. Total time: {total:.2f} seconds.\n"
        )
        training_log_file.flush()
        training_log_file.close()
        validation_log_file.close()

    @staticmethod
    def infer(
        data: str,
        model: str,
        output: str,
        **model_params,
    ) -> None:
        """
        Run inference on a key-chunked HDF5 file and write predictions into a new HDF5 file.

        This function copies ``test_data`` to an output file and adds a dataset
        ``{key}/{y_pred_dataset}`` for every top-level key.

        Parameters
        ----------
        test_data : str
            Path to the input HDF5 file. Each top-level key must contain ``x_dataset``.
        trained_model_weights : str
            Path to a PyTorch ``state_dict`` file (e.g. ``best.pth``).
        output_path : str
            Output directory where the prediction HDF5 will be written.
        add_channels : bool, optional
            If False, use only the first two channels and ``UNetPlusPlus``.
            If True, require 4 channels and use ``UNetPlusPlusRNN``.
            Default: False.
        n_classes : int, optional
            Number of output classes for ``UNetPlusPlus``. Default: 1.
            Must be 1 when ``add_channels`` is True.
        x_dataset : str, optional
            Dataset name under each key for inputs. Default: "x_0".
        y_pred_dataset : str, optional
            Dataset name under each key where predictions will be stored. Default: "y_pred".
        output_h5_name : Optional[str], optional
            Output HDF5 filename. If None, use ``<input_basename>.preds.h5``. Default: None.
        device : Optional[str], optional
            Force device string like "cuda:0" or "cpu". Default: auto-detect.

        Raises
        ------
        KeyError
            If required datasets are missing under the first key.
        """
        add_channels = False
        n_classes = 1
        x_dataset = "x_0"
        y_pred_dataset = "y_pred"
        device = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        trained_model_weights = model

        # Copy input -> output (do not modify input in-place)
        shutil.copyfile(data, output)

        with h5py.File(output, "r+") as f:
            keys = list(f.keys())
            if len(keys) == 0:
                raise ValueError(f"No keys found in HDF5 file: {out_h5}")

            k0 = keys[0]
            if x_dataset not in f[k0]:
                raise KeyError(f"Missing '{k0}/{x_dataset}' in {out_h5}")

            x0 = f[k0][x_dataset]
            channel_size = int(x0.shape[1])
            polymorphisms = int(x0.shape[3])

            # Build model
            if add_channels:
                if channel_size != 4:
                    raise ValueError(
                        f"add_channels=True expects 4 input channels, got {channel_size}."
                    )
                if int(n_classes) != 1:
                    raise ValueError(
                        "UNetPlusPlusRNN currently supports n_classes == 1 only."
                    )
                model = UNetPlusPlusRNN(polymorphisms=polymorphisms)
                input_channels = 4
            else:
                if channel_size < 2:
                    raise ValueError(
                        f"Expected at least 2 input channels, got {channel_size}."
                    )
                model = UNetPlusPlus(num_classes=int(n_classes), input_channels=2)
                input_channels = 2

            ckpt = torch.load(trained_model_weights, map_location=dev)
            model.load_state_dict(ckpt)
            model.to(dev)
            model.eval()

            for key in keys:
                x_np = np.asarray(f[key][x_dataset])[:, :input_channels].astype(
                    np.float32, copy=False
                )
                x = torch.from_numpy(x_np).to(dev)

                with torch.no_grad():
                    logits = model(x)

                # Convert logits -> probabilities
                if int(n_classes) == 1:
                    pred = torch.sigmoid(logits)  # binary prob in [0, 1]
                else:
                    pred = torch.softmax(logits, dim=1)  # multiclass prob

                # Standardize to (chunk, 1, H, W) when binary output returns (chunk, H, W)
                if pred.ndim == 3:
                    pred_np = (
                        pred.detach()
                        .cpu()
                        .numpy()[:, None, :, :]
                        .astype(np.float32, copy=False)
                    )
                elif pred.ndim == 4:
                    pred_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    raise ValueError(
                        f"Unexpected model output shape: {tuple(pred.shape)}"
                    )

                if y_pred_dataset in f[key]:
                    f[key][y_pred_dataset][...] = pred_np
                else:
                    f[key].create_dataset(
                        y_pred_dataset,
                        data=pred_np,
                        dtype=np.float32,
                        compression="lzf",
                    )

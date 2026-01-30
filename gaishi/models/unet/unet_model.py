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
from gaishi.models.unet.dataloader_h5 import (
    split_keys,
    H5BatchSpec,
    build_dataloaders_from_h5,
)


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
    ) -> None:
        """
        Train a UNet model on a key chunked HDF5 dataset and save the best weights.

        The input HDF5 file must contain multiple top level groups (keys). Each key stores
        a fixed size chunk under ``x_0`` and labels under ``y``. Training batches are formed
        by concatenating multiple key chunks along the sample axis. The number of keys per
        batch is derived from ``batch_size`` and the per key ``chunk_size`` read from the file.

        Outputs:

        1. ``best.pth``: model weights with the lowest validation loss
        2. ``training.log``: training log
        3. ``validation.log``: validation log per epoch history
        4. ``val_keys.pkl``: validation keys used for the split

        Model selection

        1. If ``add_channels`` is False, train ``UNetPlusPlus(num_classes=n_classes, input_channels=2)``
        2. If ``add_channels`` is True, train ``UNetPlusPlusRNN(polymorphisms=W)``
           and require that ``x_0`` has exactly 4 channels and ``n_classes == 1``

        Parameters
        ----------
        data : str
            Path to the HDF5 training file.
        output : str
            Path to the best weight file.
        trained_model_file : Optional[str], optional
            Path to a weights file to initialize the model before training. If None, training
            starts from random initialization. Default: None.
        add_channels : bool, optional
            If False, use only the first two channels from ``x_0``. If True, use all channels
            and select the neighbor gap fusion model. Default: False.
        n_classes : int, optional
            Number of output classes. For binary classification this is typically 1. Default: 1.
        learning_rate : float, optional
            Learning rate for Adam. Default: 0.001.
        batch_size : int, optional
            Total number of samples per optimization step after concatenation across keys.
            Must be divisible by the per key ``chunk_size`` stored in the file. Default: 32.
        label_noise : float, optional
            Noise magnitude used for label smoothing during training. Default: 0.01.
        n_early : int, optional
            Early stopping patience in epochs. Training stops after more than this many epochs
            without validation loss improvement. Default: 10.
        n_epochs : int, optional
            Maximum number of epochs. Default: 100.
        min_delta : float, optional
            Minimum required decrease in validation loss to be considered an improvement
            for early stopping and best-checkpoint saving. If the validation loss does
            not decrease by more than ``min_delta`` compared to the current best, the
            epoch is treated as "no improvement" and the early-stopping patience
            counter is incremented. Set to 0.0 to disable this threshold. Default: 1e-4
        label_smooth : bool, optional
            Whether to apply label smoothing to training labels. Default: True.

        Raises
        ------
        ValueError
            If the HDF5 file contains no keys.
        ValueError
            If ``net`` is not supported.
        ValueError
            If the training labels contain no positive class.
        ValueError
            If ``batch_size`` is not divisible by ``chunk_size``.
        ValueError
            If ``add_channels`` is True but the input does not have 4 channels.
        ValueError
            If ``add_channels`` is True but ``n_classes`` is not 1.
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

        load_file = h5py.File(data, "r")
        keys = list(load_file.keys())
        if len(keys) == 0:
            raise ValueError(f"No keys found in HDF5 file: {training_data}")

        first_key = keys[0]

        chunk_size = int(load_file[first_key]["x_0"].shape[0])
        channel_size = int(load_file[first_key]["x_0"].shape[1])
        polymorphisms = int(load_file[first_key]["x_0"].shape[3])

        if add_channels:
            input_channels = channel_size
        else:
            input_channels = 2

        # Deterministic key split for train and validation
        val_prop = 0.05
        split_seed = 0
        train_keys, val_keys = split_keys(keys, val_prop=val_prop, seed=split_seed)

        # Save validation keys for reproducibility
        pickle.dump(val_keys, open(os.path.join(output_dir, "val_keys.pkl"), "wb"))

        # Compute negative to positive ratio on training keys only
        all_counts0 = 0
        all_counts1 = 0

        for key in train_keys:
            y_ds = load_file[key]["y"][()]
            values, counts = np.unique(y_ds, return_counts=True)
            value_to_count = {int(v): int(c) for v, c in zip(values, counts)}
            all_counts0 += value_to_count.get(0, 0)
            all_counts1 += value_to_count.get(1, 0)

        if all_counts1 == 0:
            raise ValueError(
                "Training labels contain no positive class, all_counts1 is 0."
            )

        ratio = all_counts0 / all_counts1

        if add_channels:
            if channel_size != 4:
                raise ValueError(
                    f"add_channels=True expects 4 input channels in x_0, got {channel_size}."
                )
            if int(n_classes) != 1:
                raise ValueError(
                    "UNetPlusPlusRNN currently supports n_classes == 1 only."
                )
            model = UNetPlusPlusRNN(polymorphisms=polymorphisms)
        else:
            if channel_size < 2:
                raise ValueError(
                    f"Expected at least 2 input channels in x_0, got {channel_size}."
                )
            model = UNetPlusPlus(num_classes=int(n_classes), input_channels=2)

        model = model.to(device)

        if trained_model_file is not None:
            checkpoint = torch.load(trained_model_file, map_location=device)
            model.load_state_dict(checkpoint)

        # Build DataLoaders that preserve the original key chunk batching semantics
        spec = H5BatchSpec(chunk_size=chunk_size, batch_size=batch_size)
        train_loader, val_loader = build_dataloaders_from_h5(
            h5_path=data,
            train_keys=train_keys,
            val_keys=val_keys,
            channels=input_channels,
            spec=spec,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            seed=split_seed,
            train_label_smooth=label_smooth,
            train_label_noise=float(label_noise),
        )

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

                y = torch.squeeze(y)
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
        load_file.close()

    @staticmethod
    def infer(
        data: str,
        model: str,
        output: str,
        add_channels: bool = False,
        n_classes: int = 1,
        x_dataset: str = "x_0",
        y_pred_dataset: str = "y_pred",
        output_h5_name: Optional[str] = None,
        device: Optional[str] = None,
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

                # Standardize to (chunk, 1, H, W) when binary output returns (chunk, H, W)
                if logits.ndim == 3:
                    pred_np = (
                        logits.detach()
                        .cpu()
                        .numpy()[:, None, :, :]
                        .astype(np.float32, copy=False)
                    )
                elif logits.ndim == 4:
                    pred_np = (
                        logits.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                else:
                    raise ValueError(
                        f"Unexpected model output shape: {tuple(logits.shape)}"
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

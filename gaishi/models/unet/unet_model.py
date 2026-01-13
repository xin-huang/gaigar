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


import h5py
import logging
import os
import pickle
import time
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.nn import BCEWithLogitsLoss
from typing import Optional
from gaishi.models import MlModel
from gaishi.models.unet import UNetPlusPlus, UNetPlusPlusRNNNeighborGapFusion
from gaishi.registries.model_registry import MODEL_REGISTRY
from gaishi.utils.dataloader_h5 import split_keys, H5BatchSpec, build_dataloaders_from_h5


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
    """

    @staticmethod
    def train(
        training_data: str,
        model_dir: str,
        trained_model_file: Optional[str] = None,
        net: str = "default",
        add_channels: bool = False,
        n_classes: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        filter_multiplier: float = 1.0,
        label_noise: float = 0.01,
        n_early: int = 10,
        n_epochs: int = 100,
        label_smooth: bool = True,
        compute_prec_rec: bool = True,
    ) -> None:
        """
        Train a UNet model on an HDF5 dataset and save the best performing weights.
        Write outputs to disk:

            - ``{model_dir}/best.weights``: best model state dict by validation loss.
            - ``{model_dir}/train.log``: training log.
            - ``{model_dir}/training_history.csv``: training history.
            - ``{model_dir}/val_keys.pkl``: validation keys used for the split.

        The HDF5 file is expected to contain multiple top level groups (keys). Each key
        stores one chunk of samples under ``x_0`` and the corresponding labels under ``y``.
        A training batch is assembled by concatenating multiple key chunks along the
        sample axis. The number of keys per batch is derived from ``batch_size`` and the
        per key ``chunk_size`` read from the file.

        Parameters
        ----------
        training_data : str
            Path to an HDF5 file containing training data.
        model_dir : str
            Output directory for logs, history, validation keys, and best weights.
        trained_model_file : Optional[str], optional
            Path to an existing model weights file to initialize training. Defaults to None.
        net : str, optional
            Network architecture identifier. Currently only "default" is supported. Defaults to "default".
        add_channels : bool, optional
            If True, use all channels present in ``x_0``. If False, use only the first two channels.
            Defaults to False.
        n_classes : int, optional
            Number of output classes. For binary classification this is typically 1. Defaults to 1.
        learning_rate : float, optional
            Learning rate for the Adam optimizer. Defaults to 0.001.
        batch_size : int, optional
            Total number of samples per optimization step after concatenation across keys.
            Must be divisible by the per key chunk size stored in the HDF5 file. Defaults to 32.
        filter_multiplier : float, optional
            Multiplier applied to the base number of convolutional filters in the network. Defaults to 1.0.
        label_noise : float, optional
            Noise magnitude used for label smoothing. Only applied when ``label_smooth`` is True.
            Defaults to 0.01.
        n_early : int, optional
            Early stopping patience in epochs. Training stops if validation loss does not improve
            for more than this number of epochs. Defaults to 10.
        n_epochs : int, optional
            Maximum number of training epochs. Defaults to 100.
        label_smooth : bool, optional
            Whether to apply label smoothing to training labels. Defaults to True.
        compute_prec_rec : bool, optional
            Whether to compute precision and recall scores during training and validation.
            These metrics are computed on CPU and can be expensive. Defaults to True.

        Raises
        ------
        ValueError
            If the HDF5 file contains no keys, if the net identifier is unsupported,
            if the training labels contain no positive class, or if batching constraints
            cannot be satisfied.
        """
        start_time = time.time()

        if not os.path.exists(model_dir):
            os.system("mkdir -p {}".format(model_dir))

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
            print(f"CUDA is available: version {torch.version.cuda}. Training on GPU ...")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Training on CPU ...")

        log_file = open(os.path.join(model_dir, "{}.log".format("train")), "w")
        log_file.write("\n")

        load_file = h5py.File(training_data, "r")
        keys = list(load_file.keys())
        if len(keys) == 0:
            raise ValueError(f"No keys found in HDF5 file: {training_data}")

        first_key = keys[0]

        print(f"Shape of input entries: {load_file[first_key]['x_0'].shape}")
        chunk_size = int(load_file[first_key]["x_0"].shape[0])
        channel_size = int(load_file[first_key]["x_0"].shape[1])

        if add_channels:
            input_channels = channel_size
        else:
            input_channels = 2

        # Deterministic key split for train and validation
        val_prop = 0.05
        split_seed = 0
        train_keys, val_keys = split_keys(keys, val_prop=val_prop, seed=split_seed)

        # Save validation keys for reproducibility
        pickle.dump(val_keys, open(os.path.join(model_dir, "val_keys.pkl"), "wb"))

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
            raise ValueError("Training labels contain no positive class, all_counts1 is 0.")

        ratio = all_counts0 / all_counts1
        print(f"negative to positive ratio in training data set: {ratio}")

        # Initialize model
        if net == "default":
            print(f"default model with {input_channels} input channels trained")
            model = UNetPlusPlus(
                int(n_classes),
                input_channels,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        else:
            raise ValueError(f"Unknown net architecture: {net}")

        print(model, file=log_file, flush=True)
        model = model.to(device)

        if trained_model_file is not None:
            checkpoint = torch.load(trained_model_file, map_location=device)
            model.load_state_dict(checkpoint)

        # Build DataLoaders that preserve the original key chunk batching semantics
        spec = H5BatchSpec(chunk_size=chunk_size, batch_size=batch_size)
        train_loader, val_loader = build_dataloaders_from_h5(
            h5_path=training_data,
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

        lr_scheduler = None
        min_val_loss = np.inf
        early_count = 0

        history = {
            "epoch": [],
            "loss": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_time": [],
        }

        print("training...")
        for epoch_ix in range(int(n_epochs)):
            t0 = time.time()

            model.train()
            losses = []
            accuracies = []
            precisions = []
            recalls = []

            for step_ix, (x, y) in enumerate(train_loader):
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

                if compute_prec_rec:
                    precisions.append(precision_score(y_bin.flatten(), y_pred_bin.flatten()))
                    recalls.append(recall_score(y_bin.flatten(), y_pred_bin.flatten()))

                if (step_ix + 1) % 1000 == 0:
                    logging.info(
                        "root: Epoch {0}, step {3}: got loss of {1}, acc: {2}".format(
                            epoch_ix, np.mean(losses), np.mean(accuracies), step_ix + 1
                        )
                    )
                    print(
                        "root: Epoch {0}, step {3}: got loss of {1}, acc: {2}".format(
                            epoch_ix, np.mean(losses), np.mean(accuracies), step_ix + 1
                        ),
                        file=log_file,
                        flush=True,
                    )
                    if compute_prec_rec:
                        print(
                            "and precision: "
                            + str(np.mean(precisions))
                            + ", recall: "
                            + str(np.mean(recalls)),
                            file=log_file,
                            flush=True,
                        )

            model.eval()
            val_losses = []
            val_accs = []
            val_precisions = []
            val_recalls = []

            for _, (x, y) in enumerate(val_loader):
                with torch.no_grad():
                    y = torch.squeeze(y)

                    x = x.to(device)
                    y = y.to(device)

                    y_pred = model(x)
                    loss = criterion(y_pred, y)

                    y_pred_bin = np.round(expit(y_pred.detach().cpu().numpy().flatten()))
                    y_bin = np.round(y.detach().cpu().numpy().flatten())

                    val_accs.append(accuracy_score(y_bin.flatten(), y_pred_bin.flatten()))
                    val_losses.append(loss.detach().item())

                    if compute_prec_rec:
                        val_precisions.append(precision_score(y_bin.flatten(), y_pred_bin.flatten()))
                        val_recalls.append(recall_score(y_bin.flatten(), y_pred_bin.flatten()))

            val_loss = float(np.mean(val_losses))

            logging.info(
                "root: Epoch {0}, got val loss of {1}, acc: {2} ".format(
                    epoch_ix, val_loss, np.mean(val_accs)
                )
            )
            print(
                "root: Epoch {0}, got val loss of {1}, acc: {2} ".format(
                    epoch_ix, val_loss, np.mean(val_accs)
                ),
                file=log_file,
                flush=True,
            )

            if compute_prec_rec:
                print(
                    "and valprecision: "
                    + str(np.mean(val_precisions))
                    + ", valrecall: "
                    + str(np.mean(val_recalls)),
                    file=log_file,
                    flush=True,
                )

            history["epoch"].append(epoch_ix)
            history["loss"].append(float(np.mean(losses)))
            history["val_loss"].append(float(np.mean(val_losses)))

            e_time = time.time() - t0
            history["epoch_time"].append(e_time)
            history["val_acc"].append(float(np.mean(val_accs)))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                logging.info("saving weights...")
                torch.save(model.state_dict(), os.path.join(model_dir, "{0}.weights".format("best")))
                early_count = 0
            else:
                early_count += 1
                if early_count > int(n_early):
                    break

            if lr_scheduler is not None:
                lr_scheduler.step()

            df = pd.DataFrame(history)
            df.to_csv(os.path.join(model_dir, "{}_history.csv".format("training")), index=False)

        total = time.time() - start_time
        log_file.write("training complete! \n")
        log_file.write("training took {} seconds... \n".format(total))
        log_file.close()
        load_file.close()

    @staticmethod
    def infer():
        pass

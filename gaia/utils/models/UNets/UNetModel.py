# Copyright 2024 Xin Huang
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


import h5py, logging, os, pickle, time, torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from scipy.special import expit
from sklearn.metrics import accuracy_score, recall_score, precision_score

from gaia.utils.generators import H5UDataGenerator
from gaia.utils.models import MLModel

from .UNetLayers import NestedUNet, NestedUNetLSTM_fwbw, NestedUNetExtraPos


class UNetModel(MLModel):
    """ """

    @staticmethod
    def train(
        weights,
        ifile,
        odir,
        net="default",
        n_classes=1,
        pickle_load=False,
        learning_rate=0.001,
        batch_size=32,
        filter_multiplier=1,
        label_noise=0.01,
        n_early=10,
        n_epochs=100,
        label_smooth=True,
        polymorphisms=128,
        compute_prec_rec=True,
    ) -> None:
        """ """
        start_time = time.time()

        if not os.path.exists(odir):
            print("create the odir")
            os.system("mkdir -p {}".format(odir))

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
            print(
                f"CUDA is available: version {torch.version.cuda}. Training on GPU ..."
            )
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Training on CPU ...")

        log_file = open(os.path.join(odir, "{}.log".format("train")), "w")
        # config.write(log_file)
        log_file.write("\n")

        # net architecture is selected, 'default' indicates the net from the intronets-paper
        if net == "default":
            model = NestedUNet(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "attention":
            model = NestedUNetAttention(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "attention_multi":
            model = NestedUNetAttention(
                int(n_classes),
                3,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "attention_multi_fwbw":
            model = NestedUNetAttention(
                int(n_classes),
                4,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "attentionblock":
            model = NestedUNetAttentionBlock(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "attentionblock":
            model = NestedUNetAttentionBlock(
                int(n_classes),
                3,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "attentionblock_multi_fwbw":
            model = NestedUNetAttentionBlock(
                int(n_classes),
                4,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "multi":
            model = NestedUNet(
                int(n_classes),
                3,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "multi_fwbw":
            model = NestedUNet(
                int(n_classes),
                4,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )
        elif net == "lstm":
            model = NestedUNetLSTM(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                small=False,
                polymorphisms=polymorphisms,
            )
        elif net == "gru":
            model = NestedUNetLSTM(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                create_gru=True,
                small=False,
                polymorphisms=polymorphisms,
            )
        elif net == "lstm_fwbw":
            model = NestedUNetLSTM_fwbw(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                small=False,
                polymorphisms=polymorphisms,
            )
        elif net == "gru_fwbw":
            model = NestedUNetLSTM_fwbw(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                create_gru=True,
                small=False,
                polymorphisms=polymorphisms,
            )
        elif net == "extra":
            model = NestedUNetExtraPos(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                small=False,
                polymorphisms=polymorphisms,
            )
        else:
            model = NestedUNet(
                int(n_classes),
                2,
                filter_multiplier=float(filter_multiplier),
                small=False,
            )

        print(model, file=log_file, flush=True)
        model = model.to(device)

        if weights is not None:
            checkpoint = torch.load(weights, map_location=device)
            model.load_state_dict(checkpoint)

        # not fully implemented yet - and probably not necessary, because pickled numpy arrays are preferably not used
        if pickle_load == True:
            pickle_file = ifile
            file = open(pickle_file, "rb")

            # dump information to that file
            dataset = pickle.load(file)

            # compute ratio
            dataset_intro = [i[1] for i in dataset]

            values, counts = np.unique(dataset_intro, return_counts=True)
            ratio = counts[0] / counts[1]
            print("pos neg ratio")
            print(ratio)

            train_size = int(0.95 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(
                dataset, [train_size, test_size]
            )

            train_loader = DataLoader(
                [i[0:2] for i in train_dataset],
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
            )
            valid_loader = DataLoader(
                [i[0:2] for i in valid_dataset],
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
            )

        else:

            load_file = h5py.File(ifile, "r")
            keys = list(load_file.keys())

            # all_values = []
            all_counts0 = 0
            all_counts1 = 0
            # computation of the positive negative ratio - for large files it is perhaps not necessary to check each key/window ('law of large numbers')
            for key in keys:
                dataset_intro = load_file[key]["y"]

                values, counts = np.unique(dataset_intro, return_counts=True)

                count0 = counts[0]
                if len(counts) > 1:
                    count1 = counts[1]
                else:
                    count1 = 0

                all_counts0 = all_counts0 + count0
                all_counts1 = all_counts1 + count1
            ratio = all_counts0 / all_counts1
            print("pos neg ratio")
            print(ratio)

            generator = H5UDataGenerator(
                h5py.File(ifile, "r"),
                batch_size=batch_size,
                label_noise=float(label_noise),
                label_smooth=label_smooth,
            )
            val_keys = generator.val_keys

            print(set(val_keys).intersection(generator.keys))

            # save them for later
            pickle.dump(val_keys, open(os.path.join(odir, "val_keys.pkl"), "wb"))

            l = generator.length
            vl = generator.val_length

            criterion = BCEWithLogitsLoss(
                pos_weight=torch.FloatTensor([ratio]).to(device)
            )
            optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

            # main part
            lr_scheduler = None

            min_val_loss = np.inf
            early_count = 0

            history = dict()
            history["epoch"] = []
            history["loss"] = []
            history["val_loss"] = []
            history["val_acc"] = []
            history["epoch_time"] = []

            # training commences
            print("training...")
            for ix in range(int(n_epochs)):
                t0 = time.time()

                model.train()

                losses = []
                accuracies = []

                precisions = []
                recalls = []

                for ij in range(l):
                    optimizer.zero_grad()
                    x, y = generator.get_batch()

                    y = torch.squeeze(y)

                    x = x.to(device)
                    y = y.to(device)

                    y_pred = model(x)

                    loss = criterion(y_pred, y)  # ${loss_change}
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                    # compute accuracy in CPU with sklearn

                    # THIS IS PROBABLY VERY TIMECONSUMING AND SLOWS DOWN THE PROCESS - one should compare the results
                    y_pred = np.round(expit(y_pred.detach().cpu().numpy().flatten()))
                    y = np.round(y.detach().cpu().numpy().flatten())

                    # append metrics for this epoch
                    accuracies.append(accuracy_score(y.flatten(), y_pred.flatten()))

                    # precision and recall during the training is nice, but at least if computed on CPU very expensive
                    if compute_prec_rec == True:
                        train_precision_score = precision_score(
                            y.flatten(), y_pred.flatten()
                        )
                        train_recall_score = recall_score(y.flatten(), y_pred.flatten())

                        precisions.append(train_precision_score)
                        recalls.append(train_recall_score)

                    if (ij + 1) % 1000 == 0:
                        logging.info(
                            "root: Epoch {0}, step {3}: got loss of {1}, acc: {2}".format(
                                ix, np.mean(losses), np.mean(accuracies), ij + 1
                            )
                        )
                        print(
                            "root: Epoch {0}, step {3}: got loss of {1}, acc: {2}".format(
                                ix, np.mean(losses), np.mean(accuracies), ij + 1
                            ),
                            file=log_file,
                            flush=True,
                        )
                        if compute_prec_rec == True:
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

                val_recalls = []
                val_precisions = []

                for step in range(vl):
                    with torch.no_grad():
                        x, y = generator.get_val_batch()

                        y = torch.squeeze(y)

                        x = x.to(device)
                        y = y.to(device)

                        y_pred = model(x)

                        loss = criterion(y_pred, y)
                        # compute accuracy in CPU with sklearn
                        y_pred = np.round(
                            expit(y_pred.detach().cpu().numpy().flatten())
                        )
                        y = np.round(y.detach().cpu().numpy().flatten())

                        # append metrics for this epoch
                        val_accs.append(accuracy_score(y.flatten(), y_pred.flatten()))
                        val_losses.append(loss.detach().item())

                        # see above: however, at least for validation it would be good to know precision and recall
                        if compute_prec_rec == True:

                            val_precision_score = precision_score(
                                y.flatten(), y_pred.flatten()
                            )
                            val_recall_score = recall_score(
                                y.flatten(), y_pred.flatten()
                            )

                            val_precisions.append(val_precision_score)
                            val_recalls.append(val_recall_score)

                val_loss = np.mean(val_losses)

                logging.info(
                    "root: Epoch {0}, got val loss of {1}, acc: {2} ".format(
                        ix, val_loss, np.mean(val_accs)
                    )
                )

                print(
                    "root: Epoch {0}, got val loss of {1}, acc: {2} ".format(
                        ix, val_loss, np.mean(val_accs)
                    ),
                    file=log_file,
                    flush=True,
                )

                if compute_prec_rec == True:
                    print(
                        "and valprecision: "
                        + str(np.mean(val_precisions))
                        + ", valrecall: "
                        + str(np.mean(val_recalls)),
                        file=log_file,
                        flush=True,
                    )

                history["epoch"].append(ix)
                history["loss"].append(np.mean(losses))
                history["val_loss"].append(np.mean(val_losses))

                e_time = time.time() - t0

                history["epoch_time"].append(e_time)
                history["val_acc"].append(np.mean(val_accs))

                val_loss = np.mean(val_losses)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    logging.info("saving weights...")
                    torch.save(
                        model.state_dict(),
                        os.path.join(odir, "{0}.weights".format("best")),
                    )

                    early_count = 0
                else:
                    early_count += 1

                    # early stop criteria
                    if early_count > int(n_early):
                        break

                if lr_scheduler is not None:
                    lr_scheduler.step()

                generator.on_epoch_end()

                df = pd.DataFrame(history)
                df.to_csv(
                    os.path.join(odir, "{}_history.csv".format("training")), index=False
                )

            # benchmark the time to train
            total = time.time() - start_time
            log_file.write("training complete! \n")
            log_file.write("training took {} seconds... \n".format(total))
            log_file.close()

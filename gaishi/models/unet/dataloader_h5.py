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


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class H5BatchSpec:
    """
    Batch specification for key-based HDF5 datasets.

    Each HDF5 key stores a fixed-size chunk of `chunk_size` samples. A training batch is
    constructed by selecting `n_keys_per_batch` keys and concatenating their chunks
    along the sample axis, yielding a total of `batch_size` samples:

        batch_size = chunk_size * n_keys_per_batch

    This class provides a single place to encode and validate this relationship.
    """

    chunk_size: int
    batch_size: int

    @property
    def n_keys_per_batch(self) -> int:
        if self.batch_size % self.chunk_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by chunk_size ({self.chunk_size})."
            )
        return self.batch_size // self.chunk_size


class H5KeyChunkDataset(Dataset):
    """
    PyTorch Dataset for key indexed, chunk based HDF5 training data.

    Each item corresponds to a single HDF5 group (a "key") and returns a fixed size
    chunk of samples stored under that key.

    Expected HDF5 layout
    --------------------
    For each key in `keys`, the HDF5 file is expected to contain at least:

    - ``{key}/{x_dataset}``: input tensor with shape
      ``(chunk_size, n_channels, n_individuals, n_polymorphisms)``.
    - ``{key}/{y_dataset}`` (optional): label tensor with shape compatible with the model,
      commonly ``(chunk_size, 1, n_individuals, n_polymorphisms)``.

    This dataset slices the channel dimension as ``x[:, :channels, ...]`` so that
    the caller can choose between using only the base channels or including extra
    feature channels (for example neighbor gap channels).

    Notes
    -----
    - The HDF5 file is opened lazily on first access. This avoids sharing an h5py
      handle across DataLoader worker processes.
    - With ``num_workers > 0``, each worker process should get its own dataset
      instance and therefore its own file handle. This pattern is commonly used
      with h5py. Avoid sharing one Dataset instance across processes manually.
    - ``x`` is returned as a NumPy array with dtype ``x_dtype``. Conversion to
      ``torch.Tensor`` is performed in the collate function.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    keys : Sequence[str]
        HDF5 group names to use as dataset items.
    channels : int, optional
        Number of input channels to read from ``x_dataset`` by taking the first
        ``channels`` channels. Defaults to 2.
    require_labels : bool, optional
        If True, require that ``{key}/{y_dataset}`` exists for every key.
        If False, missing labels return ``None``. Defaults to True.
    x_dataset : str, optional
        Dataset name under each key for inputs. Defaults to "x_0".
    y_dataset : str, optional
        Dataset name under each key for labels. Defaults to "y".
    x_dtype : np.dtype, optional
        NumPy dtype used for returned input arrays. Defaults to ``np.int32``.

    Returns
    -------
    (x, y) : Tuple[np.ndarray, Optional[np.ndarray]]
        - x : Input array with shape ``(chunk_size, channels, n_individuals, n_polymorphisms)``.
        - y : Label array, or None if labels are absent and ``require_labels`` is False.
    """

    def __init__(
        self,
        h5_path: str,
        keys: Sequence[str],
        channels: int = 2,
        require_labels: bool = True,
        x_dataset: str = "x_0",
        y_dataset: str = "y",
        x_dtype: np.dtype = np.int32,
    ) -> None:
        self.h5_path = h5_path
        self.keys = list(keys)
        self.channels = int(channels)
        self.require_labels = bool(require_labels)
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.x_dtype = x_dtype

        self._h5: Optional[h5py.File] = None

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            # Each Dataset instance (and each worker process) opens its own handle.
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        key = self.keys[idx]
        h5 = self._get_h5()

        x = np.asarray(h5[key][self.x_dataset])[:, : self.channels].astype(
            self.x_dtype, copy=False
        )

        if self.y_dataset in h5[key]:
            y = np.asarray(h5[key][self.y_dataset])
        else:
            if self.require_labels:
                raise KeyError(f"Missing '{self.y_dataset}' under key '{key}'.")
            y = None

        return x, y


def make_h5_collate_fn(
    *,
    label_smooth: bool = False,
    label_noise: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Callable[
    [List[Tuple[np.ndarray, Optional[np.ndarray]]]],
    Tuple[torch.Tensor, Optional[torch.Tensor]],
]:
    """
    Create a DataLoader collate function for key-chunked HDF5 samples.

    The Dataset is expected to yield items of the form ``(x, y)``, where each item
    corresponds to one HDF5 key and contains a chunk of samples:

    - ``x``: ``(chunk_size, channels, n_individuals, n_polymorphisms)``
    - ``y``: ``(chunk_size, 1, n_individuals, n_polymorphisms)`` or compatible shape,
      or ``None`` if labels are absent.

    The returned collate function concatenates all ``x`` arrays (and all non-None
    ``y`` arrays) along axis 0 to form a single batch tensor. This matches the
    "key-based" batching pattern where a DataLoader batch groups multiple keys and
    the final batch size equals the sum of their chunk sizes.

    Label smoothing
    --------------
    If ``label_smooth`` is True, labels are perturbed elementwise as:

    ``Y_smooth = Y * (1 - e) + 0.5 * e``

    where ``e ~ Uniform(0, label_noise)``. This is intended for training only;
    validation labels should remain unchanged.

    Parameters
    ----------
    label_smooth : bool, optional
        Whether to apply label smoothing to labels in the batch. Defaults to False.
    label_noise : float, optional
        Maximum noise value for the uniform distribution used in label smoothing.
        Defaults to 0.01.
    rng : np.random.Generator, optional
        RNG used for label smoothing. If None, a new default RNG is created.

    Returns
    -------
    Callable
        A collate function ``collate(batch) -> (x_out, y_out)`` where:
        - ``x_out`` is a ``torch.FloatTensor`` produced from concatenated inputs.
        - ``y_out`` is a ``torch.FloatTensor`` from concatenated labels, or ``None``
          if no labels were provided in the batch.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _collate(
        batch: List[Tuple[np.ndarray, Optional[np.ndarray]]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for x, y in batch:
            xs.append(x)
            if y is not None:
                ys.append(y)

        x_out = torch.from_numpy(
            np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        )

        if len(ys) == 0:
            return x_out, None

        y_np = np.concatenate(ys, axis=0).astype(np.float32, copy=False)

        if label_smooth:
            # y = y*(1-e) + 0.5*e  with e~U(0, label_noise)
            e = rng.uniform(0.0, float(label_noise), size=y_np.shape).astype(
                np.float32, copy=False
            )
            y_np = y_np * (1.0 - e) + 0.5 * e

        y_out = torch.from_numpy(y_np)
        return x_out, y_out

    return _collate


def build_dataloaders_from_h5(
    *,
    h5_path: str,
    train_keys: Sequence[str],
    val_keys: Sequence[str],
    channels: int,
    spec: H5BatchSpec,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 0,
    train_label_smooth: bool = True,
    train_label_noise: float = 0.01,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders for a key chunked HDF5 dataset.

    Batching semantics
    ------------------
    Each dataset item corresponds to one HDF5 key and returns a chunk of
    ``spec.chunk_size`` samples. The DataLoader batch size is set to
    ``spec.n_keys_per_batch``, meaning each DataLoader batch contains that many
    keys. The collate function concatenates these chunks, producing tensors with
    sample dimension equal to ``spec.batch_size``:

    ``spec.batch_size = spec.chunk_size * spec.n_keys_per_batch``.

    This preserves the common pattern used in key chunked HDF5 training pipelines.

    Reproducibility
    ---------------
    This function uses NumPy RNGs derived from ``seed`` for label smoothing only.
    If you need deterministic DataLoader shuffling across runs, you should also
    control PyTorch randomness, for example by setting ``torch.manual_seed(seed)``
    or by passing a seeded ``generator`` to the DataLoader.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    train_keys : Sequence[str]
        Keys used for training.
    val_keys : Sequence[str]
        Keys used for validation.
    channels : int
        Number of input channels to read from ``x_0`` by slicing
        ``x[:, :channels, ...]``.
    spec : H5BatchSpec
        Chunk size and target batch size specification. Determines how many keys
        are grouped per batch.
    num_workers : int, optional
        Number of DataLoader worker processes. Defaults to 0.
    pin_memory : bool, optional
        Whether to pin memory for faster host to GPU transfers. Defaults to True.
    seed : int, optional
        Seed used to initialize NumPy RNGs for label smoothing. Defaults to 0.
    train_label_smooth : bool, optional
        Whether to apply label smoothing to training labels in the train loader.
        Defaults to True.
    train_label_noise : float, optional
        Noise strength used for training label smoothing. Defaults to 0.01.

    Returns
    -------
    (train_loader, val_loader) : Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        - train_loader yields ``(x, y)`` with ``x`` as float32 tensor and training
          label smoothing optionally applied.
        - val_loader yields ``(x, y)`` with labels unchanged.
    """
    # Separate RNGs so train/val behavior is deterministic
    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)

    train_ds = H5KeyChunkDataset(
        h5_path=h5_path,
        keys=train_keys,
        channels=channels,
        require_labels=True,
    )
    val_ds = H5KeyChunkDataset(
        h5_path=h5_path,
        keys=val_keys,
        channels=channels,
        require_labels=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=spec.n_keys_per_batch,  # number of keys per batch
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=make_h5_collate_fn(
            label_smooth=train_label_smooth,
            label_noise=train_label_noise,
            rng=train_rng,
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=spec.n_keys_per_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=make_h5_collate_fn(
            label_smooth=False,  # never smooth validation labels
            rng=val_rng,
        ),
    )

    return train_loader, val_loader


def split_keys(
    keys: Sequence[str],
    val_prop: float = 0.05,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Split HDF5 group keys into training and validation subsets.

    The split is deterministic given the same input `keys` and `seed`. Keys are
    shuffled using a NumPy random generator and then partitioned such that the
    first ``int(len(keys) * val_prop)`` keys form the validation set and the
    remaining keys form the training set.

    Parameters
    ----------
    keys : Sequence[str]
        Iterable of HDF5 group names (each key typically corresponds to one data chunk).
    val_prop : float, optional
        Fraction of keys assigned to validation. Defaults to 0.05.
    seed : int, optional
        Seed for the random number generator used to shuffle keys. Defaults to 0.

    Returns
    -------
    (train_keys, val_keys) : Tuple[List[str], List[str]]
        Two lists containing the training keys and validation keys, respectively.

    Notes
    -----
    - The split is performed at the key level. All samples within a key are kept together.
    - If ``val_prop`` is too small relative to the number of keys, ``n_val`` may be 0.
    """
    keys = list(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = int(len(keys) * float(val_prop))
    val_keys = keys[:n_val]
    train_keys = keys[n_val:]

    return train_keys, val_keys

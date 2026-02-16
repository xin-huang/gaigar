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


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Callable, List, Optional, Sequence, Tuple


class H5Dataset(Dataset):
    """
    PyTorch dataset for a unified, replicate-indexed HDF5 schema.

    Each item corresponds to one replicate/window `r` and returns `(x, y)`.

    The HDF5 file is expected to contain:

    - Input arrays under `/data`:
      - `/data/Ref_genotype` : array, shape (R, N, L)
      - `/data/Tgt_genotype` : array, shape (R, N, L)
      - `/data/Gap_to_prev`  : array, shape (R, N, L)  (required if `channels=4`)
      - `/data/Gap_to_next`  : array, shape (R, N, L)  (required if `channels=4`)

    - Optional labels under `/targets`:
      - `/targets/Label` : array, shape (R, N, L)

    where:
    - `R` is the number of replicates/windows,
    - `N` is the number of haplotypes/rows (or samples-by-ploidy, depending on your encoding),
    - `L` is the number of sites/positions per window.

    Notes
    -----
    - The HDF5 handle is opened lazily on the first access and cached in the dataset
      instance. If you use `num_workers > 0` in a DataLoader, each worker process
      will have its own dataset instance after forking/spawn; this pattern is commonly
      used with HDF5, but you may still want to close handles explicitly in tests.
    - Labels are returned with a leading singleton channel dimension: `(1, N, L)`.

    Parameters
    ----------
    h5_file : str
        Path to the HDF5 file.
    indices : Optional[Sequence[int]], default=None
        Subset of replicate indices to expose. If None, the dataset spans
        all replicates `0..R-1`.
    channels : int, default=4
        Number of input channels to return. Must be 2 or 4.

        - If 2: returns `[Ref_genotype, Tgt_genotype]`
        - If 4: returns `[Ref_genotype, Tgt_genotype, Gap_to_prev, Gap_to_next]`
    require_labels : bool, default=True
        If True, raises a KeyError when `/targets/Label` is absent. If False,
        returns `y=None` when labels are missing.
    x_dtype : numpy.dtype, default=numpy.int32
        Output dtype for `x` after stacking.

    Returns
    -------
    x : numpy.ndarray
        Input tensor as a NumPy array of shape `(C, N, L)` where `C` is `channels`.
    y : Optional[numpy.ndarray]
        Label array of shape `(1, N, L)` if present, otherwise None when
        `require_labels=False`.

    Raises
    ------
    ValueError
        If `channels` is not 2 or 4.
    KeyError
        If `/targets/Label` is missing and `require_labels=True`.
    """

    def __init__(
        self,
        h5_file: str,
        indices: Optional[Sequence[int]] = None,
        channels: int = 4,  # 2 or 4
        require_labels: bool = True,
        x_dtype: np.dtype = np.int32,
    ) -> None:
        self.h5_file = h5_file
        self.indices = None if indices is None else list(map(int, indices))
        self.channels = int(channels)
        if self.channels not in (2, 4):
            raise ValueError("channels must be 2 or 4")
        self.require_labels = bool(require_labels)
        self.x_dtype = x_dtype

        self._h5: Optional[h5py.File] = None
        self._n_total: Optional[int] = None

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r")
        return self._h5

    def _get_n_total(self) -> int:
        if self._n_total is None:
            h5 = self._get_h5()
            self._n_total = int(h5["/data/Ref_genotype"].shape[0])
        return self._n_total

    def __len__(self) -> int:
        return self._get_n_total() if self.indices is None else len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h5 = self._get_h5()
        r = int(idx) if self.indices is None else int(self.indices[idx])

        ref = np.asarray(h5["/data/Ref_genotype"][r])
        tgt = np.asarray(h5["/data/Tgt_genotype"][r])

        if self.channels == 2:
            x = np.stack([ref, tgt], axis=0)
        else:
            gp = np.asarray(h5["/data/Gap_to_prev"][r])
            gn = np.asarray(h5["/data/Gap_to_next"][r])
            x = np.stack([ref, tgt, gp, gn], axis=0)

        x = x.astype(self.x_dtype, copy=False)

        if "/targets/Label" in h5:
            y = np.asarray(h5["/targets/Label"][r])[None, :, :]  # (1,N,L)
        else:
            if self.require_labels:
                raise KeyError("Missing /targets/Label in this HDF5 file.")
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
    Build a DataLoader `collate_fn` for H5Dataset samples.

    This collate function:
    - stacks per-sample `x` arrays into a float32 tensor of shape `(B, C, N, L)`,
    - stacks labels into a float32 tensor of shape `(B, 1, N, L)` when present,
    - optionally applies per-element label smoothing/noise.

    Label smoothing (when enabled) samples `e ~ Uniform(0, label_noise)` and applies:
    `y <- y * (1 - e) + 0.5 * e`

    This keeps labels in `[0, 1]` and nudges them toward 0.5, with magnitude bounded
    by `label_noise`.

    Parameters
    ----------
    label_smooth : bool, default=False
        Whether to apply label smoothing/noise to the stacked labels.
    label_noise : float, default=0.01
        Maximum noise amplitude used when `label_smooth=True`.
    rng : Optional[numpy.random.Generator], default=None
        RNG used to sample smoothing noise. If None, a new default generator
        is created.

    Returns
    -------
    collate_fn : Callable
        A function compatible with PyTorch DataLoader that maps
        `List[(x, y)] -> (X, Y)` where:

        - `X` is a torch.float32 tensor of shape `(B, C, N, L)`.
        - `Y` is a torch.float32 tensor of shape `(B, 1, N, L)` if any labels are
          present in the batch; otherwise None.

    Notes
    -----
    - If a batch contains a mixture of labeled and unlabeled samples, only the
      labeled samples are stacked into `Y`. (In typical usage, batches are uniform.)
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

        x_np = np.stack(xs, axis=0).astype(np.float32, copy=False)  # (B,C,N,L)
        x_out = torch.from_numpy(x_np)

        if len(ys) == 0:
            return x_out, None

        y_np = np.stack(ys, axis=0).astype(np.float32, copy=False)  # (B,1,N,L)

        if label_smooth:
            e = rng.uniform(0.0, float(label_noise), size=y_np.shape).astype(
                np.float32, copy=False
            )
            y_np = y_np * (1.0 - e) + 0.5 * e

        return x_out, torch.from_numpy(y_np)

    return _collate


def build_dataloaders_from_h5(
    *,
    h5_file: str,
    channels: int,
    batch_size: int,
    val_prop: float = 0.05,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 0,
    train_label_smooth: bool = True,
    train_label_noise: float = 0.01,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """
    Construct train/validation DataLoaders from an HDF5 file via `torch.random_split`.

    This function creates a base `H5Dataset` spanning all replicates/windows in
    the HDF5 file and splits it into training and validation subsets using
    `torch.utils.data.random_split` with a seeded generator for determinism.

    Training batches can optionally apply label smoothing in the collate function,
    while validation batches do not apply smoothing.

    Parameters
    ----------
    h5_file : str
        Path to the HDF5 file.
    channels : int
        Number of input channels (2 or 4). Passed to `H5Dataset`.
    batch_size : int
        Batch size for both loaders.
    val_prop : float, default=0.05
        Proportion of replicates assigned to validation.
        The validation size is `int(n_total * val_prop)`.
    num_workers : int, default=0
        Number of DataLoader worker processes.
    pin_memory : bool, default=True
        Whether to enable pinned memory in the DataLoaders.
    seed : int, default=0
        Seed for the deterministic split (PyTorch generator). Also used to seed
        NumPy RNGs for label smoothing (`seed` for train, `seed + 1` for val).
    train_label_smooth : bool, default=True
        Whether to apply label smoothing/noise in the training collate function.
    train_label_noise : float, default=0.01
        Maximum smoothing noise amplitude used when `train_label_smooth=True`.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training loader built from the training subset. Uses `shuffle=True` and
        `drop_last=True`.
    val_loader : torch.utils.data.DataLoader
        Validation loader built from the validation subset. Uses `shuffle=False`
        and `drop_last=True`.
    train_indices : list[int]
        Replicate indices (into the base dataset) assigned to training.
        This is `train_subset.indices` from the `Subset` returned by `random_split`.
    val_indices : list[int]
        Replicate indices assigned to validation (`val_subset.indices`).

    Notes
    -----
    - `random_split` returns `Subset` objects whose `.indices` refer to indices
      in the base dataset (which in this case correspond to replicate IDs).
    - Shuffling order within the training loader is controlled by PyTorch; set
      `torch.manual_seed(...)` externally if you need deterministic per-epoch
      shuffling in tests.
    - Both loaders use `drop_last=True`, so incomplete final batches are dropped.
    """
    # Base dataset over all replicates/windows
    base_ds = H5Dataset(
        h5_file=h5_file,
        indices=None,
        channels=channels,
        require_labels=True,
    )

    n_total = len(base_ds)
    n_val = int(n_total * float(val_prop))
    n_train = n_total - n_val

    # Deterministic split
    g = torch.Generator()
    g.manual_seed(int(seed))
    train_ds, val_ds = random_split(base_ds, [n_train, n_val], generator=g)

    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=make_h5_collate_fn(
            label_smooth=False,
            rng=val_rng,
        ),
    )

    return train_loader, val_loader, train_ds.indices, val_ds.indices

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
import multiprocessing
import numpy as np
import pandas as pd
from typing import Any, Mapping, Sequence, Union


def write_tsv(file_name: str, data_dict: dict, lock: multiprocessing.Lock) -> None:
    """
    Write the data dictionary to a TSV file.

    Parameters
    ----------
    file_name : str
        Path to the TSV file.
    data_dict : dict
        Dictionary containing the data to be written to the TSV file.
    lock : multiprocessing.Lock
        Lock for synchronizing multiprocessing operations.
    """
    converted_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            array_list = value.tolist()
            converted_dict[key] = array_list
        else:
            converted_dict[key] = value

    df = pd.DataFrame([converted_dict])

    with lock:
        with open(file_name, "a") as f:
            df.to_csv(f, header=False, index=False, sep="\t")


def write_h5(
    h5_file: str,
    entries: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ds_type: str,
    lock: multiprocessing.Lock,
    compression: str = "lzf",
) -> None:
    """
    Write a single HDF5 file for either training input or inference input.

    The file is always overwritten (opened with mode "w"). This function writes only
    the model inputs and, for training, the supervision targets. Prediction outputs
    such as logits are not written here.

    Data are stored in a unified, slice-friendly layout where the first dimension
    indexes replicates or windows. Sample order may differ between entries due to
    per-entry sorting. Therefore, per-entry row identity is stored as integer ids
    (``/index/ref_ids`` and ``/index/tgt_ids``) that point to global sample tables
    stored once in ``/meta``.

    Parameters
    ----------
    h5_file : str
        Output HDF5 path.
    entries : Mapping[str, Any] or Sequence[Mapping[str, Any]]
        One entry or a list of entries. Each entry corresponds to one replicate
        or one inference window.

        Required keys for both ``ds_type="train"`` and ``ds_type="infer"``:
        - ``Ref_genotype`` : array_like, shape (N, L)
        - ``Tgt_genotype`` : array_like, shape (N, L)
        - ``Gap_to_prev``  : array_like, shape (N, L)
        - ``Gap_to_next``  : array_like, shape (N, L)
        - ``Ref_sample``   : sequence of length N, sample names in row order
        - ``Tgt_sample``   : sequence of length N, sample names in row order
        - ``Chromosome``   : scalar, stored in ``/meta`` attributes

        Additional required keys for ``ds_type="train"``:
        - ``Label``     : array_like, shape (N, L)
        - ``Seed``      : scalar
        - ``Replicate`` : scalar

        Additional required keys for ``ds_type="infer"``:
        - ``Position``  : array_like, shape (L,)
    ds_type : {"train", "infer"}
        Dataset type. Controls whether training targets or inference coordinates
        are written.
    lock : multiprocessing.Lock
        Inter-process lock that serializes HDF5 writes.
    compression : str, default "lzf"
        HDF5 dataset compression filter.

    Notes
    -----
    HDF5 schema written by this function.

    Common datasets
    - ``/data/Ref_genotype``  : uint32, shape (n, N, L)
    - ``/data/Tgt_genotype``  : uint32, shape (n, N, L)
    - ``/data/Gap_to_prev``   : int64,  shape (n, N, L)
    - ``/data/Gap_to_next``   : int64,  shape (n, N, L)
    - ``/index/ref_ids``      : uint32, shape (n, N)
    - ``/index/tgt_ids``      : uint32, shape (n, N)
    - ``/meta/ref_sample_table`` : utf-8 strings, shape (K_ref,)
    - ``/meta/tgt_sample_table`` : utf-8 strings, shape (K_tgt,)
    - ``/meta`` attributes: ``N``, ``L``, ``Chromosome``

    Training-only datasets (``ds_type="train"``)
    - ``/targets/Label``      : uint8,  shape (n, N, L)
    - ``/index/Seed``         : int64,  shape (n,)
    - ``/index/Replicate``    : int64,  shape (n,)

    Inference-only datasets (``ds_type="infer"``)
    - ``/coords/Position``    : int64,  shape (n, L)

    The integer row id arrays map each row in ``Ref_genotype`` and ``Tgt_genotype``
    back to the global sample tables. This preserves per-entry sorting while still
    allowing efficient slicing across entries.
    """
    if ds_type not in ("train", "infer"):
        raise ValueError('ds_type must be "train" or "infer"')

    if isinstance(entries, Mapping):
        entries_list = [entries]
    else:
        entries_list = list(entries)
    if not entries_list:
        raise ValueError("entries is empty")

    e0 = entries_list[0]

    base_required = (
        "Ref_genotype",
        "Tgt_genotype",
        "Gap_to_prev",
        "Gap_to_next",
        "Ref_sample",
        "Tgt_sample",
    )
    for k in base_required:
        if k not in e0:
            raise KeyError(f"Missing key in entry[0]: {k}")

    if ds_type == "train":
        extra_required = ("Label", "Seed", "Replicate")
    else:
        extra_required = ("Position",)
    for k in extra_required:
        if k not in e0:
            raise KeyError(f"Missing key in entry[0]: {k}")

    ref0 = np.asarray(e0["Ref_genotype"])
    tgt0 = np.asarray(e0["Tgt_genotype"])
    if ref0.ndim != 2 or tgt0.ndim != 2:
        raise ValueError("Ref_genotype and Tgt_genotype must be 2D arrays (N, L)")
    if ref0.shape != tgt0.shape:
        raise ValueError(
            f"Ref_genotype shape {ref0.shape} != Tgt_genotype shape {tgt0.shape}"
        )
    N, L = ref0.shape
    n = len(entries_list)

    ref_table, tgt_table = [], []
    ref_seen, tgt_seen = set(), set()

    for i, e in enumerate(entries_list):
        for k in base_required:
            if k not in e:
                raise KeyError(f"Entry {i}: missing {k}")

        ref = np.asarray(e["Ref_genotype"])
        tgt = np.asarray(e["Tgt_genotype"])
        if ref.shape != (N, L) or tgt.shape != (N, L):
            raise ValueError(f"Entry {i}: genotype shape mismatch, expected {(N, L)}")

        gp = np.asarray(e["Gap_to_prev"])
        gn = np.asarray(e["Gap_to_next"])
        if gp.shape != (N, L) or gn.shape != (N, L):
            raise ValueError(f"Entry {i}: gap shape mismatch, expected {(N, L)}")

        if len(e["Ref_sample"]) != N or len(e["Tgt_sample"]) != N:
            raise ValueError(
                f"Entry {i}: Ref_sample and Tgt_sample length must be N={N}"
            )

        if ds_type == "train":
            for k in extra_required:
                if k not in e:
                    raise KeyError(f"Entry {i}: missing {k}")
            y = np.asarray(e["Label"])
            if y.shape != (N, L):
                raise ValueError(f"Entry {i}: Label shape mismatch, expected {(N, L)}")
        else:
            pos = np.asarray(e["Position"])
            if pos.ndim != 1 or pos.shape[0] != L:
                raise ValueError(f"Entry {i}: Position shape mismatch, expected {(L,)}")

        for s in e["Ref_sample"]:
            name = str(s)
            if name not in ref_seen:
                ref_seen.add(name)
                ref_table.append(name)
        for s in e["Tgt_sample"]:
            name = str(s)
            if name not in tgt_seen:
                tgt_seen.add(name)
                tgt_table.append(name)

    ref_map = {name: idx for idx, name in enumerate(ref_table)}
    tgt_map = {name: idx for idx, name in enumerate(tgt_table)}

    ref_ids = np.empty((n, N), dtype=np.uint32)
    tgt_ids = np.empty((n, N), dtype=np.uint32)

    for i, e in enumerate(entries_list):
        ref_ids[i, :] = np.asarray(
            [ref_map[str(s)] for s in e["Ref_sample"]], dtype=np.uint32
        )
        tgt_ids[i, :] = np.asarray(
            [tgt_map[str(s)] for s in e["Tgt_sample"]], dtype=np.uint32
        )

    seeds = None
    reps = None
    if ds_type == "train":
        seeds = np.asarray([int(e["Seed"]) for e in entries_list], dtype=np.int64)
        reps = np.asarray([int(e["Replicate"]) for e in entries_list], dtype=np.int64)

    str_dt = h5py.string_dtype(encoding="utf-8")

    with lock:
        with h5py.File(h5_file, "w") as h5f:
            h5f.require_group("/data")
            h5f.require_group("/index")
            meta = h5f.require_group("/meta")

            if ds_type == "train":
                h5f.require_group("/targets")
            else:
                h5f.require_group("/coords")

            meta.attrs["n"] = int(n)  # number of replicates or windows
            meta.attrs["N"] = int(N)  # number of samples
            meta.attrs["L"] = int(L)  # number of sites
            meta.attrs["Chromosome"] = str(e0["Chromosome"])

            h5f.create_dataset(
                "/meta/ref_sample_table",
                data=np.asarray(ref_table, dtype=object),
                dtype=str_dt,
            )
            h5f.create_dataset(
                "/meta/tgt_sample_table",
                data=np.asarray(tgt_table, dtype=object),
                dtype=str_dt,
            )

            h5f.create_dataset(
                "/data/Ref_genotype",
                shape=(n, N, L),
                dtype=np.uint32,
                chunks=(1, N, L),
                compression=compression,
            )
            h5f.create_dataset(
                "/data/Tgt_genotype",
                shape=(n, N, L),
                dtype=np.uint32,
                chunks=(1, N, L),
                compression=compression,
            )
            h5f.create_dataset(
                "/data/Gap_to_prev",
                shape=(n, N, L),
                dtype=np.int64,
                chunks=(1, N, L),
                compression=compression,
            )
            h5f.create_dataset(
                "/data/Gap_to_next",
                shape=(n, N, L),
                dtype=np.int64,
                chunks=(1, N, L),
                compression=compression,
            )

            h5f.create_dataset(
                "/index/ref_ids",
                data=ref_ids,
                dtype=np.uint32,
                chunks=(min(64, n), N),
                compression=compression,
            )
            h5f.create_dataset(
                "/index/tgt_ids",
                data=tgt_ids,
                dtype=np.uint32,
                chunks=(min(64, n), N),
                compression=compression,
            )

            if ds_type == "train":
                h5f.create_dataset(
                    "/targets/Label",
                    shape=(n, N, L),
                    dtype=np.uint8,
                    chunks=(1, N, L),
                    compression=compression,
                )
                h5f.create_dataset(
                    "/index/Seed", data=seeds, dtype=np.int64, compression=compression
                )
                h5f.create_dataset(
                    "/index/Replicate",
                    data=reps,
                    dtype=np.int64,
                    compression=compression,
                )
            else:
                h5f.create_dataset(
                    "/coords/Position",
                    shape=(n, L),
                    dtype=np.int64,
                    chunks=(1, L),
                    compression=compression,
                )

            for i, e in enumerate(entries_list):
                h5f["/data/Ref_genotype"][i] = np.asarray(
                    e["Ref_genotype"], dtype=np.uint32
                )
                h5f["/data/Tgt_genotype"][i] = np.asarray(
                    e["Tgt_genotype"], dtype=np.uint32
                )
                h5f["/data/Gap_to_prev"][i] = np.asarray(
                    e["Gap_to_prev"], dtype=np.int64
                )
                h5f["/data/Gap_to_next"][i] = np.asarray(
                    e["Gap_to_next"], dtype=np.int64
                )

                if ds_type == "train":
                    h5f["/targets/Label"][i] = np.asarray(e["Label"], dtype=np.uint8)
                else:
                    h5f["/coords/Position"][i] = np.asarray(
                        e["Position"], dtype=np.int64
                    )

            h5f.flush()

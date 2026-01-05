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
from typing import Any, Union, Optional


def write_h5(
    file_name: str,
    entries: Union[dict[str, Any], list[dict[str, Any]]],
    lock: multiprocessing.Lock,
    *,
    stepsize: int = 192,
    is_phased: bool = True,
    chunk_size: int = 1,
    neighbor_gaps: bool = True,
    start_nr: Optional[int] = None,
    set_attributes: bool = True,
) -> int:
    """
    Append one or many entry dictionaries to an HDF5 file.

    This is a high-level convenience wrapper that prepares per-window/per-sample
    dictionaries for on-disk storage. Each input dictionary is normalized to the
    expected schema, packed into the writer-specific nested-list representation,
    and then appended to the HDF5 file using the low-level writer.

    Parameters
    ----------
    file_name : str
        Path to the output HDF5 file.
    entries : dict[str, Any] or list[dict[str, Any]]
        Either a single entry dictionary or a list of entry dictionaries.
        Each entry is expected to contain the fields required by the schema
        normalizer and packer (for example: ``Start``, ``End``, ``Ref_sample``,
        ``Tgt_sample``, ``Ref_genotype``, ``Tgt_genotype``, ``Label``,
        ``Gap_to_prev``, ``Gap_to_next``, and
        ``Replicate``).
    lock : multiprocessing.Lock
        Inter-process lock used to serialize HDF5 writes.
    stepsize : int, optional
        Window length used when an entry has ``Start == "Random"``. In that case,
        ``Start`` is set to 0 and ``End`` is set to ``stepsize``. Defaults to 192.
    is_phased : bool, optional
        Whether the sample identifiers encode phased haplotypes. If True, haplotype
        indices are preserved but converted to 0-based indexing (hap-1). If False,
        haplotype indices are set to 0. Defaults to True.
    chunk_size : int, optional
        Number of packed entries passed to the low-level writer per write call.
        Defaults to 1. Keep ``chunk_size=1`` until chunk semantics in the low-level
        writer are validated.
    neighbor_gaps : bool, optional
        Whether to include neighbor-gap information as additional feature channels in the
        stored feature tensor. When enabled, two channels are included: the distance to the
        previous variant (gap_to_prev) and the distance to the next variant (gap_to_next).
        Defaults to True.
    start_nr : int, optional
        Starting group id for writing. If None, the low-level writer determines
        the next id from the file attribute ``last_index`` (defaulting to 0 if
        missing). Defaults to None.
    set_attributes : bool, optional
        Whether to update the file attribute ``last_index`` after writing.
        Defaults to True.

    Returns
    -------
    int
        The next available group id after the final write.

    Raises
    ------
    KeyError
        If required keys are missing from an entry dictionary.

    Notes
    -----
    - This function normalizes and packs entries into the schema expected by the
      low-level HDF writer.
    - Keep `chunk_size=1` until the underlying writer’s chunk semantics are validated.
    """
    if isinstance(entries, dict):
        entries = [entries]

    packed_entries = []
    for d in entries:
        d_norm = _normalize_hdf_entry(dict(d), stepsize=stepsize, is_phased=is_phased)
        packed_entries.append(_pack_hdf_entry(d_norm))

    return _append_hdf_entries(
        hdf_file=file_name,
        input_entries=packed_entries,
        lock=lock,
        start_nr=start_nr,
        chunk_size=chunk_size,
        neighbor_gaps=neighbor_gaps,
        set_attributes=set_attributes,
    )


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


def _normalize_hdf_entry(
    data_dict: dict, stepsize: int = 192, is_phased: bool = True
) -> dict:
    """
    Normalize a single entry dictionary to match the expected HDF5 schema.

    This helper prepares a per-window/per-sample `data_dict` for downstream HDF5
    writers by enforcing a small set of schema requirements:

    - Ensures numeric start/end coordinates. If ``data_dict["Start"] == "Random"``,
      the window is forced to start at 0 and the end is set to ``stepsize``.
    - Adds a combined ``StartEnd`` field as ``[Start, End]``.
    - Converts ``Ref_sample`` and ``Tgt_sample`` from string identifiers of the
      form ``"<prefix>_<ind>_<hap>"`` into integer pairs ``[ind, hap]``.
      For phased data, haplotypes are converted to 0-based indices (hap-1).
      For unphased data, the haplotype index is always set to 0.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing (at minimum) the keys ``Start``, ``End``,
        ``Ref_sample``, and ``Tgt_sample``.
    stepsize : int, optional
        Window length used to derive ``End`` when ``Start`` is ``"Random"``.
        Defaults to 192.
    is_phased : bool, optional
        Whether the data are phased. If True, haplotype indices are preserved
        (converted to 0-based). If False, haplotype indices are set to 0.
        Defaults to True.

    Returns
    -------
    dict
        The same dictionary instance, modified in place, with normalized
        ``Start``, ``End``, ``StartEnd``, ``Ref_sample``, and ``Tgt_sample``.

    Notes
    -----
    This function mutates ``data_dict`` in place and returns it for convenience.
    """
    if data_dict["Start"] == "Random":
        data_dict["Start"] = 0
        data_dict["End"] = stepsize

    data_dict["StartEnd"] = [data_dict["Start"], data_dict["End"]]

    def _parse_sample_id(sample_id: str) -> list[int]:
        parts = sample_id.split("_")
        ind = int(parts[1])
        hap = (int(parts[2]) - 1) if is_phased else 0
        return [ind, hap]

    data_dict["Ref_sample"] = [_parse_sample_id(s) for s in data_dict["Ref_sample"]]
    data_dict["Tgt_sample"] = [_parse_sample_id(s) for s in data_dict["Tgt_sample"]]

    return data_dict


def _pack_hdf_entry(data_dict: dict) -> list:
    """
    Pack a normalized entry dictionary into the ordered nested-list format
    expected by the HDF5 writer.

    The returned structure is a list of groups, where each group is a list of
    one or more arrays/scalars in a fixed order. This order is treated as part
    of the on-disk schema and must stay consistent with the HDF5 writer.

    Parameters
    ----------
    data_dict : dict
        A single preprocessed entry. Expected to contain at least the keys:
        ``Ref_genotype``, ``Tgt_genotype``, ``Label``, ``Ref_sample``, ``Tgt_sample``,
        ``StartEnd``, ``End``, ``Replicate``, ``Position``,
        ``Gap_to_prev``, and ``Gap_to_next``.

    Returns
    -------
    list
        Nested list representation of the entry in the schema-defined order.

    Notes
    -----
    This helper does not modify ``data_dict``.
    """
    keys = (
        ("Ref_genotype", "Tgt_genotype"),
        ("Label",),
        ("Ref_sample", "Tgt_sample"),
        ("StartEnd",),
        ("End",),
        ("Replicate",),
        ("Position",),
        ("Gap_to_prev",),
        ("Gap_to_next",),
    )
    return [[data_dict[k] for k in group] for group in keys]


def _append_hdf_entries(
    hdf_file: str,
    input_entries: list,
    lock: multiprocessing.Lock,
    start_nr: Optional[int] = None,
    x_name: str = "x_0",
    y_name: str = "y",
    ind_name: str = "indices",
    pos_name: str = "pos",
    ix_name: str = "ix",
    chunk_size: int = 1,
    neighbor_gaps: bool = True,
    set_attributes: bool = True,
) -> int:
    """
    Append packed entries to an HDF5 file.

    This is a low-level writer that assumes `input_entries` are already packed into
    the nested-list format produced by `_pack_hdf_entry`. Each entry is written
    under an integer group id (e.g., ``"0/"``, ``"1/"``), and the next available
    id is tracked in the file attribute ``last_index`` when `start_nr` is None.

    When `neighbor_gaps` is True, two additional feature channels are appended to the
    feature tensor (gap to the previous variant and gap to the next variant). This function
    asserts that these channels are integer-valued and have shapes compatible with the
    feature tensor.

    Parameters
    ----------
    hdf_file : str
        Path to the output HDF5 file.
    input_entries : list
        A list of packed entries. Each packed entry must follow the order defined
        by `_pack_hdf_entry`, and must provide:
        - entry[0]: base feature channels (e.g., ref/tgt genotype channels)
        - entry[1]: label tensor
        - entry[2]: indices tensor (reference/target sample ids)
        - entry[3]: position tensor (e.g., StartEnd)
        - entry[-2], entry[-1]: neighbor-gap channels (gap to previous variant, gap to next variant;
                                only if `neighbor_gaps` is True)
        - entry[5]: replicate/index value for `ix`
    lock : multiprocessing.Lock
        Inter-process lock to serialize HDF5 writes.
    start_nr : int, optional
        Starting group id. If None, the value is read from the file attribute
        ``last_index`` (defaulting to 0 if missing).
    x_name, y_name, ind_name, pos_name, ix_name : str, optional
        Dataset names created within each group for features, labels, indices,
        positions, and the replicate/index tracker, respectively.
    chunk_size : int, optional
        Number of entries written per loop iteration. The current implementation
        is designed to be used with ``chunk_size=1``. Other values are accepted
        but require careful validation of index semantics.
    neighbor_gaps : bool, optional
        If True, append two neighbor-gap feature channels to the base feature tensor
        before writing: the gap to the previous variant and the gap to the next variant.
    set_attributes : bool, optional
        If True, update the file attribute ``last_index`` to the next available
        group id after writing.

    Returns
    -------
    int
        The next available group id after the final write.
    """
    additional_x_features = 2 if neighbor_gaps else 0

    with lock:
        with h5py.File(hdf_file, "a") as h5f:
            if start_nr is None:
                start_nr = h5f.attrs.get("last_index", 0)

            for i in range(0, len(input_entries) - chunk_size + 1, chunk_size):
                entry = input_entries[i]
                group_id = start_nr + i

                act_shape0, act_shape1, act_shape2, act_shape3 = (
                    np.array(entry[0]).shape,
                    np.array(entry[1]).shape,
                    np.array(entry[2]).shape,
                    np.array(entry[3]).shape,
                )

                dset1 = h5f.create_dataset(
                    f"{i + start_nr}/{x_name}",
                    (
                        chunk_size,
                        act_shape0[0] + additional_x_features,
                        act_shape0[1],
                        act_shape0[2],
                    ),
                    compression="lzf",
                    dtype=np.uint32,
                )
                dset2 = h5f.create_dataset(
                    f"{i + start_nr}/{y_name}",
                    (chunk_size, 1, act_shape1[1], act_shape1[2]),
                    compression="lzf",
                    dtype=np.uint8,
                )
                dset3 = h5f.create_dataset(
                    f"{i + start_nr}/{ind_name}",
                    (chunk_size, act_shape2[0], act_shape2[1], act_shape2[2]),
                    compression="lzf",
                    dtype=np.uint32,
                )
                dset4 = h5f.create_dataset(
                    f"{i + start_nr}/{pos_name}",
                    (chunk_size, 1, act_shape3[0], act_shape3[1]),
                    compression="lzf",
                    dtype=np.uint32,
                )
                dset5 = h5f.create_dataset(
                    f"{i + start_nr}/{ix_name}",
                    (chunk_size, 1, 1),
                    compression="lzf",
                    dtype=np.uint32,
                )

                for k in range(chunk_size):
                    entry = input_entries[i + k]

                    if neighbor_gaps:
                        features = np.concatenate([entry[0], entry[-2], entry[-1]])
                    else:
                        features = entry[0]

                    dset1[k] = features
                    dset2[k] = entry[1]
                    dset3[k] = entry[2]
                    dset4[k] = entry[3]
                    dset5[k] = entry[5]

            if set_attributes:
                h5f.attrs["last_index"] = group_id + 1
                h5f.flush()

            return group_id + 1

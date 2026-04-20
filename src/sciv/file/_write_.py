# -*- coding: UTF-8 -*-

import os.path
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import Literal, Optional

import h5py
import pandas as pd
from pandas import DataFrame
from anndata import AnnData

from .. import util as ul
from ..util import path, to_sparse, chrtype, check_adata_get

__name__: str = "file_write"

_Field = Optional[Literal['real', 'complex', 'pattern', 'integer']]


def save_h5ad(data: AnnData, file: path) -> AnnData:
    """
    Save AnnData data to h5ad file.

    Parameters
    ----------
    data : AnnData
        Input AnnData object to save.
    file : path
        Path to save file.

    Returns
    -------
    AnnData
        The input AnnData object.
    """
    ul.log(__name__).info("Saving data to {}".format(file))
    return data.write_h5ad(Path(file), compression='gzip')


def save_h5(data: dict, save_file: path, group_name: str = "matrix") -> None:
    """
    Save H5 data to H5 file.

    Parameters
    ----------
    data : dict
        Input H5 data to save.
    save_file : path
        Input path to save file.
    group_name: str, default="matrix"
        The group name.
    
    Returns
    -------
    H5 file
        The input H5 file.
    """
    h5_dict = dict(data)

    file = h5py.File(f"{str(save_file)}", 'w')
    grp = file.create_group(group_name)

    for key, value in h5_dict.items():
        grp.create_dataset(key, data=value)

    file.close()


def save_pkl(data, save_file: path, is_verbose: bool = False) -> None:
    """
    Save pkl data to pkl file.
    
    Parameters
    ----------
    data : any
        Input data to save.
    save_file : path
        Input path to save file.
    is_verbose: Set true to print log;
    
    Returns
    -------
    pkl file
        The input pkl file.
    """
    if is_verbose:
        ul.log(__name__).info("Saving data to {}".format(save_file))

    with open(str(save_file), 'wb') as f:
        pickle.dump(data, f)  # type: ignore


def to_meta(
    adata: AnnData,
    dir_path: path,
    layer: str = None,
    feature_name: str = "peaks.bed",
    field: _Field = None
) -> None:
    """
    Convert AnnData object into metadata directory containing matrix, feature files, etc.

    This function exports single-cell data into standard 10x Genomics format, including:
    - matrix.mtx: Sparse matrix file in Matrix Market format
    - annotation.txt: Cell annotation information
    - barcodes.tsv: Cell barcodes list
    - peaks.bed or specified feature file: Genomic feature information

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing single-cell data.
    dir_path : path
        Output directory path for storing generated metadata files.
    layer : str, optional
        layer: The layer of data that needs to form meta files;
        If None, uses adata.X as the main data matrix.
    feature_name : str, default="peaks.bed"
        Output name for the feature file. If starts with "peaks", 
        feature indices will be parsed by chromosome position into BED format.
    field : _Field, optional
        Matrix data type field, available values:
        - 'real': Real numbers
        - 'complex': Complex numbers
        - 'pattern': Pattern matrix (no values)
        - 'integer': Integer values
        If None, automatically determined from data type.

    Returns
    -------
    Directory
        The input directory.
    """

    dir_path = str(dir_path)
    ul.file_method(__name__).makedirs(dir_path)

    # Convert dense matrices to sparse matrices
    sparse_matrix = to_sparse(adata.layers[layer] if layer is not None else adata.X)
    # write mtx file
    ul.log(__name__).info(f"Write mtx file")
    import scipy.io as scio
    scio.mmwrite(os.path.join(dir_path, 'matrix.mtx'), sparse_matrix.T, field=field)

    # Cell annotation
    ul.log(__name__).info(f"Write cell annotation")
    cell_info: DataFrame = adata.obs
    cell_info["barcodes"] = adata.obs.index.to_list()
    cell_info.to_csv(
        os.path.join(dir_path, "annotation.txt"),
        index=False,
        sep="\t",
        lineterminator="\n",
        encoding="utf-8"
    )

    # barcodes
    ul.log(__name__).info(f"Write barcodes")
    barcodes = pd.DataFrame(adata.obs.index.to_list(), columns=["index"])
    barcodes.to_csv(
        os.path.join(dir_path, "barcodes.tsv"),
        index=False,
        header=False,
        sep="\t",
        lineterminator="\n",
        encoding="utf-8"
    )

    # feature
    ul.log(__name__).info(f"Write feature")
    feature_info: DataFrame = adata.var
    if feature_name.split(".")[0] == "peaks":
        feature = pd.DataFrame(feature_info.index.to_list(), columns=["index"])
        new_feature = feature["index"].astype(str).str.split("[:-]", expand=True)
        new_feature.to_csv(
            os.path.join(dir_path, feature_name),
            index=False,
            header=False,
            sep="\t",
            lineterminator="\n",
            encoding="utf-8"
        )
    else:
        feature = pd.DataFrame(feature_info.index.to_list(), columns=["index"])
        feature.to_csv(
            os.path.join(dir_path, feature_name),
            index=False,
            header=False,
            sep="\t",
            lineterminator="\n",
            encoding="utf-8"
        )


def to_fragments(
    adata: AnnData,
    fragments: str,
    layer: str = None,
    is_sort: bool = True,
    is_gz: bool = True,
    is_keep: bool = False
) -> None:
    """
    Convert AnnData format data into fragments format file.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing single-cell data.
    fragments : str
        Output file path for the fragments file.
    layer : str, optional
        The layer of data to use for generating fragments file.
        If None, uses the main data matrix (adata.X).
    is_sort : bool, default=True
        Whether to sort the output by chromosome and start position.
        Sorts chromosomes in natural order (chr1, chr2, ..., chrX, chrY, chrM).
    is_gz : bool, default=True
        Whether to compress the output file using gzip.
        Uses pysam.tabix_compress for compression.
    is_keep : bool, default=False
        Whether to keep the uncompressed fragments file after compression.
        Only effective when is_gz is True. If False, the uncompressed
        file is deleted after successful compression.
    
    Returns
    -------
    None
        Writes fragments file to the specified path.

    Examples
    --------
    >>> adata = sciv.dl.read_sc_atac_file()
    >>> sciv.fl.to_fragments(adata, "/path/pbmc_fragments.tsv")

    Note
    --------
    When exporting results processed with SnapATAC2, using the snapatac2.ex.export_fragments function is not
    recommended.
    """

    output_path = os.path.dirname(fragments)

    if output_path != '':
        ul.file_method(__name__).makedirs(output_path)

    if is_gz:
        try:
            import pysam
        except ImportError:
            raise ImportError(
                "The 'pysam' package is required for gzip compression. "
                "Please install it using: pip install pysam."
            )

    data = check_adata_get(adata=adata, layer=layer, is_dense=False, is_matrix=False).T

    # get group information
    data_obs: DataFrame = data.obs.copy()
    data_var: DataFrame = data.var.copy()

    if "chr" not in data_obs.columns or "start" not in data_obs.columns or "end" not in data_obs.columns:
        ul.log(__name__).error("`chr` or `start`or  `end` not in obs column")
        raise ValueError("`chr` or `start` or `end` not in obs column")

    if "barcodes" not in data_var.columns:
        ul.log(__name__).error(f"`barcodes` not in obs column")
        raise ValueError(f"`barcodes` not in obs column")

    if is_sort:
        ul.log(__name__).info("Sort chromatin")
        data_obs["chr"] = data_obs["chr"].astype(chrtype)
        data_obs.sort_values(["chr", "start"], inplace=True)
        source_row_size = data_obs.shape[0]
        data_obs.dropna(subset=['chr'])

        if source_row_size > data_obs.shape[0]:
            chrs_str = ",".join(list(chrtype.categories))
            ul.log(__name__).warning(f"The chromatin with `chr` not in `{chrs_str}` has been deleted here.")

        data = data[data_obs.index, :]

    matrix = to_sparse(data.X, is_matrix=False)

    row_size, col_size = data.shape
    row_range, col_range = range(row_size), range(col_size)

    # Convert to dictionary
    barcodes_dict: dict = dict(zip(list(col_range), data_var.index))
    peaks_dict: dict = dict(zip(list(row_range), zip(data_obs["chr"], data_obs["start"], data_obs["end"])))

    nonzero = matrix.nonzero()
    nonzero_size = nonzero[0].size
    ul.log(__name__).info(f"Get size {row_size, col_size} ===> nonzero size: {nonzero_size}")

    ul.log(__name__).info(f"Generate the `fragments` file {fragments}.")
    with open(fragments, mode="w", encoding="utf-8", newline="\n") as f:

        f.write(f"# output_file = {fragments}\n")
        f.write(f"# layer = {layer}\n")
        f.write(f"# features: {row_size}, barcodes: {col_size}, nonzero: {nonzero_size}\n")

        for row, col in tqdm(zip(nonzero[0], nonzero[1])):
            # info
            peaks = peaks_dict[row]
            barcodes = barcodes_dict[col]
            f.write(f"{peaks[0]}\t{peaks[1]}\t{peaks[2]}\t{barcodes}\t{matrix[row, col]}\n")

    if is_gz:
        is_success: bool = True

        try:
            ul.log(__name__).info(f"Generate the `fragments` file {fragments}.gz.")
            import pysam
            pysam.tabix_compress(fragments, f"{fragments}.gz", force=True)
        except Exception as e:
            is_success: bool = False
            ul.log(__name__).error(f"Compression of {fragments} file failed. \n {e}")

        if is_success and not is_keep:
            os.remove(fragments)

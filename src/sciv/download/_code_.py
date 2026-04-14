# -*- coding: UTF-8 -*-

import os
import tarfile
from typing import Tuple

from anndata import AnnData
from pandas import DataFrame

from .. import util as ul

__name__: str = "download_core"

from ..file import read_sc_atac, read_variants, read_h5ad

file_method = ul.file_method(__name__, is_verbose=True)


def _download_core_(filename: str, is_force: bool = False) -> None:
    """
    Core function to download a file from the remote server to the local cache.

    Parameters
    ----------
    filename : str
        The name of the file to download.
    is_force : bool, optional
        If True, force re-download even if the file exists. Default is False.
    """
    cache_path = ul.project_cache_path
    file_method.makedirs(cache_path)

    download_file = f"https://bio.liclab.net/scvmap_static/sciv/{filename}"
    file_method.download_file(download_file, os.path.join(cache_path, filename), is_force=is_force)


def download_sc_atac_file(is_force: bool = False) -> None:
    """
    Download the scATAC2 file from the remote server to the local cache.

    Parameters
    ----------
    is_force : bool, optional
        If True, force re-download even if the file exists. Default is False.

    Examples
    --------
    >>> sciv.dl.download_sc_atac_file()
    """
    _download_core_("GSE139369_ELM_sim_snapATAC2.h5ad", is_force)


def download_trait_file(is_force: bool = False) -> None:
    """
    Download the trait file from the remote server to the local cache.

    Parameters
    ----------
    is_force : bool, optional
        If True, force re-download even if the file exists. Default is False.

    Examples
    --------
    >>> sciv.dl.download_trait_file()
    """
    _download_core_("example_traits.tar.gz", is_force)

    file_path = os.path.join(ul.project_cache_path, "example_traits.tar.gz")

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=ul.project_cache_path)


def download_trs_file(is_force: bool = False) -> None:
    """
    Download the TRS file from the remote server to the local cache.

    Parameters
    ----------
    is_force : bool, optional
        If True, force re-download even if the file exists. Default is False.

    Examples
    --------
    >>> sciv.dl.download_trs_file()
    """
    _download_core_("trs.h5ad", is_force)


def download_trs_score_file(is_force: bool = False) -> None:
    """
    Download the TRS score file from the remote server to the local cache.

    Parameters
    ----------
    is_force : bool, optional
        If True, force re-download even if the file exists. Default is False.

    Examples
    --------
    >>> sciv.dl.download_trs_score_file()
    """
    _download_core_("trs_method_score.h5ad", is_force)


def read_sc_atac_file() -> AnnData:
    """
    Read the scATAC-seq file from the local cache.

    Returns
    -------
    AnnData
        The scATAC-seq data.

    Examples
    --------
    >>> sciv.dl.read_sc_atac_file()
    """
    cache_path = ul.project_cache_path
    file_path: str = os.path.join(cache_path, "GSE139369_ELM_sim_snapATAC2.h5ad")

    if not os.path.exists(file_path):
        download_sc_atac_file()

    return read_sc_atac(file_path)


def read_trait_file() -> Tuple[dict, DataFrame]:
    """
    Read the trait files from the local cache.

    Returns
    -------
    Tuple[dict, DataFrame]
        The trait data.

    Examples
    --------
    >>> sciv.dl.read_trait_file()
    """
    cache_path = ul.project_cache_path
    file_path: str = os.path.join(cache_path, "example_traits")

    if not os.path.exists(file_path):
        download_trait_file()

    variant_column_map: dict = {0: "chr", 1: "position", 3: "rsId", 4: "pp"}
    return read_variants(file_path, column_map=variant_column_map)


def read_trs_file() -> AnnData:
    """
    Read the TRS file from the local cache.

    Returns
    -------
    AnnData
        The TRS data.

    Examples
    --------
    >>> sciv.dl.read_trs_file()
    """
    cache_path = ul.project_cache_path
    file_path: str = os.path.join(cache_path, "trs.h5ad")

    if not os.path.exists(file_path):
        download_trs_file()

    return read_h5ad(file_path)


def read_trs_score_file() -> AnnData:
    """
    Read the TRS score file from the local cache.

    Returns
    -------
    AnnData
        The TRS score data.

    Examples
    --------
    >>> sciv.dl.read_trs_score_file()
    """
    cache_path = ul.project_cache_path
    file_path: str = os.path.join(cache_path, "trs_method_score.h5ad")

    if not os.path.exists(file_path):
        download_trs_score_file()

    return read_h5ad(file_path)

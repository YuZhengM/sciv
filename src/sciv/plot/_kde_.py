# -*- coding: UTF-8 -*-

from typing import Literal, Any

import numpy as np
from anndata import AnnData
import seaborn as sns
from matplotlib.figure import Figure
from tqdm import tqdm

from .. import util as ul
from ..tool import down_sampling_data
from ..util import path, check_adata_get, plot_end, plot_start

__name__: str = "plot_kde"

log = ul.log(__name__, "ERROR")


def kde(
    adata: AnnData,
    layer: str = None,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    axis: Literal[-1, 0, 1] = -1,
    sample_number: int = 1000000,
    is_legend: bool = True,
    output: path = None,
    show: bool = False,
    close: bool = False,
    **kwargs: Any
) -> tuple[Figure, Any]:
    """
    Plot Kernel Density Estimation (KDE) for single-cell data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (rows) and variables (columns).
    layer : str, optional
        Which layer of `adata` to use. If None, uses `adata.X`.
    x_name : str, optional
        Label for the x-axis.
    y_name : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the plot.
    axis : Literal[-1, 0, 1], default=-1
        Axis along which to compute KDE:
        - -1: Flatten all data and compute single KDE.
        - 0: Compute KDE for each column (variable).
        - 1: Compute KDE for each row (observation).
    sample_number : int, default=1000000
        Maximum number of samples to use for KDE computation.
        If data exceeds this, random downsampling is applied.
    is_legend : bool, default=True
        Whether to display legend when axis is 0 or 1.
    output : path, optional
        Path to save the figure. If None, figure is not saved.
    show : bool, default=True
        Whether to display the figure.
    close : bool, default=False
        Whether to close the figure after displaying.
    **kwargs : Any
        Additional keyword arguments passed to `seaborn.kdeplot`.
    """
    log.info("Start plotting the Kernel density estimation chart")

    fig, ax = plot_start()

    data = check_adata_get(adata, layer=layer, is_dense=True, is_matrix=False)

    sns.set_theme(style="whitegrid")

    # Random sampling
    if axis == -1:
        matrix = down_sampling_data(data.X, sample_number)
        sns.kdeplot(matrix, fill=True)
    elif axis == 0:
        col_number = data.shape[1]
        if data.shape[0] * data.shape[1] > sample_number:
            row_number: int = sample_number // col_number

            for i in tqdm(range(col_number)):
                _vector_ = down_sampling_data(data.X[:, i], row_number)
                sns.kdeplot(np.array(_vector_).flatten(), fill=True, **kwargs)
        else:
            for i in tqdm(range(col_number)):
                sns.kdeplot(np.array(data.X[:, i]).flatten(), fill=True, **kwargs)

        if is_legend:
            ax.legend(list(adata.var.index))

    elif axis == 1:
        row_number = data.shape[0]
        if data.shape[0] * data.shape[1] > sample_number:
            col_number: int = sample_number // row_number

            for i in tqdm(range(row_number)):
                _vector_ = down_sampling_data(data.X[i, :], col_number)
                sns.kdeplot(np.array(_vector_).flatten(), fill=True, **kwargs)
        else:
            for i in tqdm(range(row_number)):
                sns.kdeplot(np.array(data.X[i, :]).flatten(), fill=True, **kwargs)

        if is_legend:
            ax.legend(list(adata.obs.index))

    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, ax

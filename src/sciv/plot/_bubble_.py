# -*- coding: UTF-8 -*-

from typing import Any

import numpy as np
import seaborn as sns
from pandas import DataFrame

from ..util import path, plot_end, plot_start

__name__: str = "plot_bubble"


def bubble(
    df: DataFrame,
    x: str,
    y: str,
    hue: str = None,
    size: str = None,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    width: float = 2,
    height: float = 2,
    bottom: float = 0,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
):
    """
    Create a bubble plot using seaborn's relplot.

    Parameters
    ----------
    df : DataFrame
        Input data structure.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    hue : str, optional
        Column name for color encoding.
    size : str, optional
        Column name for size encoding.
    x_name : str, optional
        Custom label for x-axis.
    y_name : str, optional
        Custom label for y-axis.
    title : str, optional
        Plot title.
    width : float, default=2
        Figure width in inches.
    height : float, default=2
        Figure height in inches.
    bottom : float, default=0
        Bottom margin adjustment.
    output : path, optional
        File path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    close : bool, default=False
        Whether to close the figure after display.
    **kwargs : Any
        Additional arguments passed to seaborn.relplot.
    """
    fig, ax = plot_start(width, height, bottom, output, show)

    if size is not None:
        _size_ = df[size].values
        sizes = (np.array(_size_).min(), np.array(_size_).max())
    else:
        sizes = None

    sns.relplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        size=size,
        sizes=sizes,
        alpha=.5,
        palette="muted",
        height=6,
        **kwargs
    )

    plot_end(fig, title, x_name, y_name, output, show, close)

# -*- coding: UTF-8 -*-

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pandas import DataFrame

from .. import util as ul
from ..util import path, type_50_colors, type_20_colors, plot_end

__name__: str = "plot_barcode"

log = ul.log(__name__, "ERROR")


def barcode(
    df: DataFrame,
    groupby: str,
    sort_column: str = "value",
    element_column: str = "id",
    width: float = 1,
    height: float = 3,
    title: str = None,
    cmap: str = "Oranges",
    bar_label: str = "TRS",
    is_ticks: bool = True,
    colors: list = None,
    ground_true: list = None,
    output: path = None,
    show: bool = False,
    close: bool = True
) -> tuple[Figure, Any, Any]:
    """
    Plot barcode plot.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    sort_column : str, optional
        Sort column.
    groupby : str, optional
        Column name for clusters.
    width : float, optional
        Width.
    height : float, optional
        Height.
    element_column : str, optional
        Trait column name.
    title : str, optional
        Title.
    cmap : str, optional
        Cmap.
    bar_label : str, optional
        Bar label.
    is_ticks : bool, optional
        Whether to show ticks.
    colors : list, optional
        Colors.
    ground_true : list, optional
        Ground true.
    output : path, optional
        Output path.
    show : bool, optional
        Whether to display the plot.
    close : bool, optional
        Whether to close the figure after display.
    """
    # sort
    df_sort = df.sort_values([element_column, sort_column], ascending=False)

    # set index
    class_list = df_sort[groupby].unique().tolist()
    id_list = df_sort[element_column].unique().tolist()
    df_sort["class_index"] = np.zeros(df_sort.shape[0])

    if colors is None:
        colors = type_20_colors if len(class_list) <= 20 else type_50_colors

    groupby_list = df_sort[groupby].unique().tolist()

    for i in class_list:

        if ground_true is not None:
            ground_true: list
            df_sort.loc[df_sort[df_sort[groupby] == i].index, ["class_index"]] = ground_true.count(i)
        else:
            df_sort.loc[df_sort[df_sort[groupby] == i].index, ["class_index"]] = groupby_list.index(i)

    class_index = np.array(df_sort["class_index"])

    # figure
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.99, bottom=0.01)

    plt.axis("off")

    gs = GridSpec(20, 20)
    ax1 = fig.add_subplot(gs[:17, 11:14] if is_ticks else gs[:18, 11:14])
    ax2 = fig.add_subplot(gs[:17, :8] if is_ticks else gs[:18, :8])

    # span the whole figure
    ax1.set_axis_off()
    ax1.imshow(
        np.array(class_index).reshape(int(df_sort.shape[0] / len(id_list)), -1),
        cmap=ListedColormap(colors),
        aspect='auto',
        interpolation='nearest'
    )
    # ax1.tick_params(axis='x', rotation=90)

    ax2.set_axis_off()
    im2 = ax2.imshow(
        np.array(df_sort[sort_column]).reshape(int(df_sort.shape[0] / len(id_list)), -1),
        cmap=cmap,
        aspect='auto',
        interpolation='nearest'
    )

    # [left, bottom, width, height]
    cax = fig.add_axes((0.12, 0.09, 0.5, 0.04) if is_ticks else (0.1, 0.04, 0.5, 0.04))
    color_bar = plt.colorbar(im2, ax=ax2, cax=cax, label=bar_label, orientation='horizontal')
    color_bar.set_label(bar_label)

    ticks = np.linspace(round(df_sort[sort_column].min(), 2), round(df_sort[sort_column].max() - 0.05, 2), 3)
    color_bar.set_ticks(ticks if is_ticks else [])

    plot_end(fig, title, output=output, show=show, close=close)

    return fig, ax1, ax2

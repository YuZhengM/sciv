# -*- coding: UTF-8 -*-

from typing import Union, Any

import numpy as np
from pandas import DataFrame

from .. import util as ul
from ..util import path, collection, get_real_predict_label, type_20_colors, type_50_colors, plot_start, plot_end

__name__: str = "plot_pie"

log = ul.log(__name__, "ERROR")


def pie(
    values: list,
    labels: list,
    title: str = None,
    pct_distance: float = 0.6,
    label_distance: float = 1.1,
    colors: list = None,
    autopct: str = '%1.2f%%',
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Create a basic pie chart with customizable parameters.

    This function generates a simple pie chart using matplotlib, with support for
    custom colors, labels, and various display options.

    Parameters
    ----------
    values : list
        The values to be plotted in the pie chart.
    labels : list
        The labels corresponding to each value in the pie chart.
    title : str, optional
        The title of the pie chart. Default is None.
    pct_distance : float, optional
        The distance of the percentage labels from the center of the pie.
        Default is 0.6.
    label_distance : float, optional
        The distance of the labels from the center of the pie. Default is 1.1.
    colors : list, optional
        A list of colors to use for the pie slices. If None, default colors
        will be used. Default is None.
    autopct : str, optional
        The format string for the percentage labels. Default is '%1.2f%%'.
    output : path, optional
        The file path to save the figure. If None, the figure will not be saved.
        Default is None.
    show : bool, optional
        Whether to display the figure. Default is True.
    close : bool, optional
        Whether to close the figure after displaying. Default is False.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib's pie function.
    """
    fig, ax = plot_start()

    size = len(values)

    if size is not len(labels):
        log.error(f"The parameter lengths of `values`({size}) and `labels`({len(labels)}) must be equal.")
        raise ValueError(f"The parameter lengths of `values`({size}) and `labels`({len(labels)}) must be equal.")

    if colors is None:
        colors = type_20_colors[:len(labels)] if size <= 20 else type_50_colors[:len(labels)]

    ax.set_axis_off()

    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=autopct,
        labeldistance=label_distance,
        pctdistance=pct_distance,
        **kwargs
    )

    ax.axis('off')

    plot_end(title, output=output, show=show, close=close)

    return ax


def pie_label(
    df: DataFrame,
    map_groupby: Union[str, collection],
    value: str = "value",
    groupby: str = "clusters",
    title: str = None,
    radius: float = 0.6,
    fontsize: float = 17,
    pct_distance: float = 0.6,
    label_distance: float = 1.1,
    colors: list = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Create a donut-style pie chart showing cluster label distribution.

    This function generates a pie chart with a central hole (donut chart) to visualize
    the distribution of predicted cluster labels against true labels. The chart displays
    the percentage of correctly predicted labels in the center.

    Parameters
    ----------
    df : DataFrame
        The input data containing cluster and value information.
    map_groupby : Union[str, collection]
        The mapping of clusters, can be a column name or a collection of cluster labels.
    value : str, optional
        The column name for values in the DataFrame. Default is "value".
    groupby : str, optional
        The column name for cluster labels in the DataFrame. Default is "clusters".
    title : str, optional
        The title of the pie chart. Default is None.
    radius : float, optional
        The radius of the inner white circle to create donut effect. Default is 0.6.
    fontsize : float, optional
        The font size for the percentage text in the center. Default is 17.
    pct_distance : float, optional
        The distance of the percentage labels from the center of the pie.
        Default is 0.6.
    label_distance : float, optional
        The distance of the labels from the center of the pie. Default is 1.1.
    colors : list, optional
        A list of colors to use for the pie slices. If None, default colors
        will be used. Default is None.
    output : path, optional
        The file path to save the figure. If None, the figure will not be saved.
        Default is None.
    show : bool, optional
        Whether to display the figure. Default is True.
    close : bool, optional
        Whether to close the figure after displaying. Default is False.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib's pie function.
    """
    fig, ax = plot_start()

    # judge
    df_columns = list(df.columns)

    if value not in df_columns:
        log.error(
            f"The `value` ({value}) parameter must be in the `df` parameter data column name ({df_columns})")
        raise ValueError(
            f"The `value` ({value}) parameter must be in the `df` parameter data column name ({df_columns})"
        )

    df_sort, cluster_size, cluster_list = get_real_predict_label(
        df=df,
        map_groupby=map_groupby,
        groupby=groupby,
        value=value
    )

    # top value
    top_predict_cluster = list(df_sort["true_label"])[:cluster_size]
    top_x = [top_predict_cluster.count(1), top_predict_cluster.count(0)]

    if colors is None:
        colors = type_20_colors[:2]

    top_sum = np.array(top_x).sum()

    ax.set_axis_off()
    ax.pie(
        top_x,
        labels=[", ".join(cluster_list), "Other"],
        colors=colors,
        startangle=90,
        labeldistance=label_distance,
        pctdistance=pct_distance,
        wedgeprops=dict(linewidth=0),
        **kwargs
    )
    ax.pie(
        [np.array(top_x).sum()],
        colors=['white'],
        radius=radius,
        startangle=90,
        wedgeprops=dict(width=radius, edgecolor='w', linewidth=0),
        **kwargs
    )
    ax.text(0, 0, "{:.2f}%".format(top_x[0] / top_sum * 100), ha='center', va='center', fontsize=fontsize)
    ax.legend(loc='upper right')

    ax.axis('off')

    plot_end(title, output=output, show=show, close=close)

    return ax

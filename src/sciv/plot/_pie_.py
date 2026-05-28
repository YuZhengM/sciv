# -*- coding: UTF-8 -*-

import os
from typing import Union, Any

import numpy as np
from matplotlib.figure import Figure
from pandas import DataFrame

from .. import util as ul
from ..util import path, collection, get_real_predict_label, type_20_colors, type_50_colors, plot_start, plot_end

__name__: str = "plot_pie"

log = ul.log(__name__, "ERROR")


def base_pie(
    values: list,
    labels: list,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    pct_distance: float = 0.6,
    label_distance: float = 1.1,
    colors: list = None,
    autopct: str = '%1.2f%%',
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> tuple[Figure, Any]:
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
    x_name : str, optional
        The label for the x-axis. Default is None.
    y_name : str, optional
        The label for the y-axis. Default is None.
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

    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, ax


def pie_label(
    df: DataFrame,
    map_groupby: Union[str, collection],
    value: str = "value",
    groupby: str = "clusters",
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    radius: float = 0.6,
    fontsize: float = 17,
    pct_distance: float = 0.6,
    label_distance: float = 1.1,
    colors: list = None,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> tuple[Figure, Any]:
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
    x_name : str, optional
        The label for the x-axis. Default is None.
    y_name : str, optional
        The label for the y-axis. Default is None.
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

    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, ax


def pie_trait(
    trait_df: DataFrame,
    trait_groupby_map: dict,
    trait_name: str = "All",
    groupby: str = "clusters",
    trait_column_name: str = "id",
    value: str = "value",
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    radius: float = 0.6,
    fontsize: float = 17,
    pct_distance: float = 0.6,
    label_distance: float = 1.1,
    colors: list = None,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
    """
    Create pie charts for trait/disease cluster distribution analysis.

    This function generates donut-style pie charts to visualize the distribution
    of trait-specific scores across different cell clusters. It supports batch
    processing for multiple traits or single trait analysis.

    Parameters
    ----------
    trait_df : DataFrame
        The input data containing trait information, cluster labels, and values.
    trait_groupby_map : dict
        A dictionary mapping trait names to their corresponding cluster mappings.
        Keys are trait names, values are cluster label mappings.
    trait_name : str, optional
        The specific trait to plot. Use "All" to plot all traits in the data.
        Default is "All".
    groupby : str, optional
        The column name for cluster labels in the DataFrame. Default is "clusters".
    trait_column_name : str, optional
        The column name for trait identifiers in the DataFrame. Default is "id".
    value : str, optional
        The column name for values/scores in the DataFrame. Default is "value".
    x_name : str, optional
        The label for the x-axis. Default is None.
    y_name : str, optional
        The label for the y-axis. Default is None.
    title : str, optional
        The base title for the pie charts. Trait name will be appended if provided.
        Default is None.
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
        The directory path to save the figures. If None, figures will not be saved.
        Default is None.
    show : bool, optional
        Whether to display the figure. Default is True.
    close : bool, optional
        Whether to close the figure after displaying. Default is False.
    **kwargs : Any
        Additional keyword arguments passed to the pie_label function.
    """
    trait_groupby_map_key_list = list(trait_groupby_map.keys())

    data: DataFrame = trait_df.copy()

    def trait_plot(trait_: str, atac_cell_df_: DataFrame) -> None:
        """
        Internal helper function to generate a pie chart for a specific trait.

        This function validates the trait exists in the mapping, filters the data
        for the specified trait, and calls pie_label to create the visualization.

        Parameters
        ----------
        trait_ : str
            The name of the trait to plot.
        atac_cell_df_ : DataFrame
            The DataFrame containing trait data for plotting.
        """
        if trait_ not in trait_groupby_map_key_list:
            log.error(f"The key in `trait_groupby_map` does not contain the `{trait_}` trait and needs to be added")
            raise ValueError(
                f"The key in `trait_groupby_map` does not contain the `{trait_}` trait and needs to be added"
            )

        log.info("Plotting pie {}".format(trait_))
        # get gene score
        trait_score = atac_cell_df_[atac_cell_df_[trait_column_name] == trait_]
        # Sort gene scores from small to large
        pie_label(
            df=trait_score[[trait_column_name, groupby, value]],
            map_groupby=trait_groupby_map[trait_],
            value=value,
            groupby=groupby,
            x_name=x_name,
            y_name=y_name,
            radius=radius,
            fontsize=fontsize,
            pct_distance=pct_distance,
            label_distance=label_distance,
            colors=colors,
            title=f"{title} {trait_}" if title is not None else title,
            output=os.path.join(output, f"cell_{trait_}_score_pie.pdf") if output is not None else None,
            show=show,
            close=close,
            **kwargs
        )

    # noinspection DuplicatedCode
    trait_list = list(set(data[trait_column_name]))
    # judge trait
    if trait_name != "All" and trait_name not in trait_list:
        log.error(f"The {trait_name} trait/disease is not in the trait/disease list {trait_list}.")
        raise ValueError(f"The {trait_name} trait/disease is not in the trait/disease list {trait_list}.")

    # plot
    if trait_name == "All":
        for trait in trait_list:
            trait_plot(trait, trait_df)
    else:
        trait_plot(trait_name, trait_df)

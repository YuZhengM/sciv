# -*- coding: UTF-8 -*-

import os
from typing import Optional, Union, Tuple, Any

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pandas import DataFrame

from .. import util as ul
from ..util import path, collection, plot_end, type_20_colors, type_50_colors

__name__: str = "plot_radar"


def radar(
    ax_x: collection,
    ax_y: collection,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    colors: collection = None,
    width: float = 4,
    height: float = 4,
    bottom: float = 0,
    center_text: str = None,
    rotation: float = 25,
    value_top: float = 0.1,
    text_top: float = 1.2,
    is_fixed: bool = False,
    is_angle: bool = True,
    y_limit: Tuple = (-0.5, 1),
    y_axis_scale: Tuple = (0, 1),
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
    """
    Plot a radar chart.

    Parameters
    ----------
    ax_x : collection
        Category labels for the radar chart.
    ax_y : collection
        Data values for each category.
    x_name : str, optional
        Label for the x-axis.
    y_name : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the chart.
    colors : collection, optional
        Colors for the radar chart.
    width : float, optional
        Width of the chart.
    height : float, optional
        Height of the chart.
    bottom : float, optional
        Bottom margin adjustment.
    center_text : str, optional
        Center text for the chart.
    rotation : float, optional
        Angle rotation for the radar chart.
    value_top : float, optional
        Value top for the radar chart.
    text_top : float, optional
        Text top for the radar chart.
    is_fixed : bool, optional
        Whether to fix the radar chart.
    is_angle : bool, optional
        Whether to use angle for the radar chart.
    y_limit : Tuple, optional
        Y-axis limit.
    y_axis_scale : Tuple, optional
        Y-axis scale.
    output : path, optional
        Output path.
    show : bool, optional
        Whether to show.
    close : bool, optional
        Whether to close.
    kwargs : Any, optional
        Keyword arguments.
    Returns
    -------
    None
    """

    if output is None and not show:
        ul.log(__name__).error(f"At least one of the `output` and `show` parameters is required")
        raise ValueError(f"At least one of the `output` and `show` parameters is required")

    fig, ax = plt.subplots(figsize=(width, height), subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(bottom=bottom)

    ax_x = list(ax_x)
    ax_y = list(ax_y)

    # Create a circular bar chart
    theta = np.linspace(0, 2 * np.pi, len(ax_x), endpoint=False).tolist()
    ax_y += ax_y[:1]
    theta += theta[:1]

    width = 2 * 2.7 / len(ax_x)

    bars = ax.bar(theta, ax_y, width=width, color=colors, edgecolor='none', alpha=0.8, zorder=3, **kwargs)

    # Add category labels
    ax.set_xticks(theta)
    ax.set_xticklabels([])

    # Set y-axis range
    ax.set_ylim(y_limit[0], y_limit[1])

    # Remove the scale value of the circle
    ax.set_yticks(np.linspace(y_axis_scale[0], y_axis_scale[1], 6))  # Set the y-axis scale position
    ax.set_yticklabels([])  # Do not display scale values
    ax.set_theta_zero_location('N')  # Set polar axis position
    ax.set_theta_direction(-1)  # The angle increases counterclockwise

    # Add numerical labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        angle = np.degrees(theta[i])
        label_position = theta[i]
        ax.text(
            label_position, height + value_top if not is_fixed else value_top, round(height, 3),
            ha='center', va='center', color='#1f1f1f', rotation=-angle + rotation if is_angle else rotation
        )

    # Add radar line
    ax.plot(theta, ax_y, color='gray', linewidth=1, zorder=1)
    # Draw radar map
    ax.fill(theta, ax_y, color='#DDDDDD', alpha=0.1, zorder=2)

    if center_text is not None:
        ax.text(0, y_limit[0], center_text, ha='center', va='center', fontsize=14, color='black', zorder=11)

    # Set the y-axis scale line color to light gray
    ax.tick_params('y', colors='#DDDDDD', grid_alpha=0.6, zorder=8)

    # Set the color of the outermost circle line
    ax.spines['polar'].set_color('#DDDDDD')

    plt.grid(axis='x', linestyle='-', alpha=0.4, zorder=9)

    # Draw peripheral category labels
    for i, label in enumerate(ax_x):
        angle = np.degrees(theta[i])
        ax.text(
            theta[i], text_top, label, ha='center', va='center', color='#1f1f1f', zorder=20,
            rotation=-angle + rotation if is_angle else rotation
        )

    # Adjust the layout to prevent label overlap
    plt.tight_layout()

    plot_end(fig, title, x_name, y_name, output, show, close)


def base_radar(
    df: DataFrame,
    ax_x: str,
    ax_y: str,
    hue: str,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    width: float = 4,
    height: float = 4,
    bottom: float = 0,
    colors: collection = None,
    line_width: float = 0.5,
    y_limit: Tuple = (0, 1),
    bbox_to_anchor: Tuple = (1.3, 1.1),
    is_fill: bool = True,
    fill_alpha: float = 0.2,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
    """
    Plot a radar chart with multiple groups.

    Parameters
    ----------
    df : DataFrame
        Input data containing the values to plot.
    ax_x : str
        Column name for category labels (x-axis categories).
    ax_y : str
        Column name for values to plot (y-axis values).
    hue : str
        Column name for grouping different lines.
    x_name : str, optional
        Label for the x-axis.
    y_name : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the chart.
    width : float, optional
        Width of the chart figure.
    height : float, optional
        Height of the chart figure.
    bottom : float, optional
        Bottom margin adjustment.
    colors : collection, optional
        Colors for each group line.
    line_width : float, optional
        Width of the radar lines.
    y_limit : Tuple, optional
        Y-axis limit range.
    bbox_to_anchor : Tuple, optional
        Position for the legend box.
    is_fill : bool, optional
        Whether to fill the radar area.
    fill_alpha : float, optional
        Transparency level for the filled area.
    output : path, optional
        Output path to save the figure.
    show : bool, optional
        Whether to display the figure.
    close : bool, optional
        Whether to close the figure after display.
    kwargs : Any, optional
        Additional keyword arguments for plotting.

    Returns
    -------
    None
    """
    if output is None and not show:
        ul.log(__name__).error(f"At least one of the `output` and `show` parameters is required")
        raise ValueError(f"At least one of the `output` and `show` parameters is required")

    fig, ax = plt.subplots(figsize=(width, height), subplot_kw=dict(polar=True))
    fig.subplots_adjust(bottom=bottom)

    ax_x_values = sorted(df[ax_x].unique())
    hue_values = sorted(df[hue].unique())

    # Calculate angle
    angles = np.linspace(0, 2 * np.pi, len(ax_x_values), endpoint=False).tolist()
    angles += angles[:1]

    if colors is None:
        colors = type_20_colors if len(hue_values) <= 20 else type_50_colors

    for i, _hue_ in enumerate(hue_values):
        k_data = df[df[hue] == _hue_]
        values = k_data[ax_y].tolist()
        values += values[:1]

        ax.plot(angles, values, color=colors[i], linewidth=line_width, label=_hue_, **kwargs)

        if is_fill:
            ax.fill(angles, values, color=colors[i], alpha=fill_alpha)

    # Set angle label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ax_x_values)

    # Set radial labels
    ax.set_rlabel_position(0)
    plt.yticks(color="grey")
    plt.ylim(y_limit[0], y_limit[1])

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, title=hue)

    plt.tight_layout()

    plot_end(fig, title, x_name, y_name, output, show, close)


def radar_trait(
    trait_df: DataFrame,
    trait_name: str = "All",
    trait_column_name: str = "id",
    value: str = "rate",
    clusters: str = "clusters",
    color: Union[collection, str] = None,
    clusters_sort: Optional[list] = None,
    width: float = 4,
    height: float = 4,
    rotation: float = 65,
    title: str = None,
    value_top: float = 0.1,
    text_top: float = 1.2,
    is_fixed: bool = False,
    is_angle: bool = True,
    y_limit: Tuple = (-0.5, 1),
    y_axis_scale: Tuple = (0, 1),
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
):
    """
    Plot radar charts for trait enrichment analysis.

    This function creates radar charts to visualize trait/disease enrichment scores
    across different clusters. It can plot either a single trait or all traits in the dataset.

    Parameters
    ----------
    trait_df : DataFrame
        Input dataframe containing trait enrichment data.
    trait_name : str, optional
        Name of the trait to plot. Use "All" to plot all traits. Default is "All".
    trait_column_name : str, optional
        Column name in trait_df that contains trait identifiers. Default is "id".
    value : str, optional
        Column name containing the enrichment values to plot. Default is "rate".
    clusters : str, optional
        Column name containing cluster identifiers. Default is "clusters".
    color : Union[collection, str], optional
        Colors for the radar chart bars. Can be a column name (str) or a collection of colors.
    clusters_sort : Optional[list], optional
        Custom order for clusters. If None, clusters are sorted by value in descending order.
    width : float, optional
        Width of the figure in inches. Default is 4.
    height : float, optional
        Height of the figure in inches. Default is 4.
    rotation : float, optional
        Rotation angle for text labels in degrees. Default is 65.
    title : str, optional
        Base title for the plot. Trait name will be appended if provided.
    value_top : float, optional
        Vertical offset for value labels above bars. Default is 0.1.
    text_top : float, optional
        Radial position for category labels. Default is 1.2.
    is_fixed : bool, optional
        If True, value labels are placed at a fixed position. Default is False.
    is_angle : bool, optional
        If True, rotate labels to align with radar angles. Default is True.
    y_limit : Tuple, optional
        Y-axis limits as (min, max). Default is (-0.5, 1).
    y_axis_scale : Tuple, optional
        Scale range for y-axis ticks as (min, max). Default is (0, 1).
    output : path, optional
        Directory path to save output PDF files. If None, files are not saved.
    show : bool, optional
        Whether to display the plot. Default is True.
    close : bool, optional
        Whether to close the figure after display. Default is False.
    kwargs : Any, optional
        Additional keyword arguments passed to the radar function.

    Returns
    -------
    None
        The function saves plots to files and/or displays them based on parameters.

    Raises
    ------
    ValueError
        If the specified trait_name is not found in the trait list.

    Examples
    --------
    >>> radar_trait(df, trait_name="Trait1", output="/path/to/output")
    >>> radar_trait(df, trait_name="All", clusters_sort=["C1", "C2", "C3"])
    """

    def trait_plot(trait_: str, cell_df_: DataFrame) -> None:
        """
        show plot
        :param trait_: trait name
        :param cell_df_:
        :return: None
        """
        ul.log(__name__).info("Plotting bar {}".format(trait_))
        trait_score = cell_df_[cell_df_[trait_column_name] == trait_]

        # Sort
        if clusters_sort is not None:
            trait_score[clusters] = pd.Categorical(trait_score[clusters], categories=clusters_sort, ordered=True)
            trait_score = trait_score.sort_values(by=clusters)
            ax_x = clusters_sort
        else:
            trait_score = trait_score.sort_values([value], ascending=False)
            ax_x = trait_score[clusters].tolist()

        colors = None
        if color is not None:
            if isinstance(color, str):
                if color in trait_score.columns:
                    colors = trait_score[color]
            elif isinstance(color, collection):
                colors = color

        radar(
            ax_x=ax_x,
            ax_y=trait_score[value].tolist(),
            title=f"{title} {trait_}" if title is not None else title,
            colors=colors,
            width=width,
            height=height,
            rotation=rotation,
            value_top=value_top,
            text_top=text_top,
            is_fixed=is_fixed,
            is_angle=is_angle,
            y_limit=y_limit,
            y_axis_scale=y_axis_scale,
            center_text=trait_,
            output=os.path.join(output, f"{trait_}_enrichment_radar.pdf") if output is not None else None,
            show=show,
            close=close,
            **kwargs
        )

    trait_list = list(set(trait_df[trait_column_name]))
    # judge trait
    if trait_name != "All" and trait_name not in trait_list:
        ul.log(__name__).error(
            f"The {trait_name} trait/disease is not in the trait/disease list {trait_list}, "
            f"Suggest modifying the {trait_column_name} parameter information"
        )
        raise ValueError(
            f"The {trait_name} trait/disease is not in the trait/disease list {trait_list}, "
            f"Suggest modifying the {trait_column_name} parameter information"
        )

    # plot
    if trait_name == "All":
        for trait in trait_list:
            trait_plot(trait_=trait, cell_df_=trait_df)
    else:
        trait_plot(trait_name, trait_df)

# -*- coding: UTF-8 -*-

from typing import Tuple, Any

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame

from .. import util as ul
from ..util import path, collection, plot_end, type_20_colors, type_50_colors, plot_start

__name__: str = "plot_radar"

log = ul.log(__name__, "ERROR")


def radar(
    ax_x: collection,
    ax_y: collection,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    colors: collection = None,
    center_text: str = None,
    rotation: float = 25,
    value_top: float = 0.1,
    text_top: float = 1.2,
    is_fixed: bool = False,
    is_angle: bool = True,
    y_limit: Tuple = (-0.5, 1),
    y_axis_scale: Tuple = (0, 1),
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Figure, Any]:
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

    fig, ax = plot_start()

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

    return fig, ax


def radar_base(
    df: DataFrame,
    ax_x: str,
    ax_y: str,
    hue: str,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    colors: collection = None,
    line_width: float = 0.5,
    y_limit: Tuple = (0, 1),
    bbox_to_anchor: Tuple = (1.3, 1.1),
    is_fill: bool = True,
    fill_alpha: float = 0.2,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Figure, Any]:
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

    fig, ax = plot_start()

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

    return fig, ax

# -*- coding: UTF-8 -*-

from typing import Optional, Tuple, Union, Any

import numpy as np
from matplotlib.lines import Line2D
from pandas import DataFrame
from anndata import AnnData
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from .. import util as ul
from ..preprocessing import adata_map_df
from ..util import path, plot_color_types, collection, plot_end, plot_start

__name__: str = "plot_line"

log = ul.log(__name__, "ERROR")


def line(
    data: Union[AnnData, DataFrame],
    x: str,
    y: str,
    layer: Optional[str] = None,
    title: Optional[str] = None,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    label: Optional[str] = None,
    legend: Optional[str] = None,
    legend_list: list = None,
    start_color_index: int = 0,
    color_step_size: int = 0,
    cmap: str = "set",
    colors: list = None,
    line_width: float = 1.5,
    x_name_rotation: float = 65,
    x_ticks: Optional[Union[int, collection]] = None,
    y_limit: Tuple[float, float] = (0, 1),
    output: Optional[path] = None,
    is_str: bool = True,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Base line plot function for visualizing data trends over time or categories.

    This function creates a line plot from either AnnData or DataFrame objects,
    supporting grouped data visualization with customizable colors, legends, and styling.

    Parameters
    ----------
    data : Union[AnnData, DataFrame]
        Input data object, can be either AnnData (single-cell data) or pandas DataFrame.
    x : str
        Column name to use for x-axis values.
    y : str
        Column name to use for y-axis values.
    layer : Optional[str], default None
        Specific layer to use from AnnData.layers when data is AnnData.
    title : Optional[str], default None
        Title of the plot.
    x_name : Optional[str], default None
        Label for x-axis. If None, uses x column name.
    y_name : Optional[str], default None
        Label for y-axis. If None, uses y column name.
    label : Optional[str], default None
        Column name used for grouping data (creates separate lines).
    legend : Optional[str], default None
        Title for the legend. If None and label is provided, uses "category".
    legend_list : list, default None
        List of specific group values to include in the plot.
    start_color_index : int, default 0
        Starting index for color selection from the color palette.
    color_step_size : int, default 0
        Step size for selecting colors from the palette.
    cmap : str, default "set"
        Type of color palette to use (key from plot_color_types).
    colors : list, default None
        Custom list of colors to use for the plot.
    line_width : float, default 1.5
        Width of the lines in the plot.
    x_name_rotation : float, default 65
        Rotation angle for x-axis tick labels (in degrees).
    x_ticks : Optional[Union[int, collection]], default None
        Custom tick positions or number of ticks for x-axis.
    y_limit : Tuple[float, float], default (0, 1)
        Y-axis limits as (min, max) tuple.
    output : Optional[path], default None
        File path to save the figure. If None, figure is not saved.
    is_str : bool, default True
        Whether to treat x-axis values as strings (affects tick formatting).
    show : bool, default True
        Whether to display the plot.
    close : bool, default False
        Whether to close the figure after display.
    **kwargs : Any
        Additional keyword arguments passed to seaborn.lineplot.
    """
    fig, ax = plot_start()

    new_data = data.copy()

    if isinstance(new_data, AnnData):

        if label is not None and legend_list is not None:
            mask = new_data.var[label].isin(legend_list)
            new_data = new_data[:, mask]

        # judge layers
        if layer is not None:

            if layer not in list(new_data.layers):
                log.error("The value of the `layer` parameter must be one of the keys in `adata.layers`.")
                raise ValueError("The value of the `layer` parameter must be one of the keys in `adata.layers`.")

            new_data.X = new_data.layers[layer]

        # DataFrame
        log.info(f"to DataFrame")
        df: DataFrame = adata_map_df(new_data, column="value")

    elif isinstance(new_data, DataFrame):

        if label is not None and legend_list is not None:
            df: DataFrame = new_data[new_data[label].isin(legend_list)].copy()
        else:
            df: DataFrame = new_data.copy()

    else:
        log.error(f"The `data` parameter only support `AnnData` and `DataFrame` class types.")
        raise ValueError(f"The `data` parameter only support `AnnData` and `DataFrame` class types.")

    if legend is None and label is not None:
        legend = "category"

    df[label] = df[label].astype(str)

    if label is not None:

        df[legend] = df[label].copy()
        hue_types = df[legend].unique().tolist()
        new_data_columns = list(df.columns)

        # noinspection DuplicatedCode
        if colors is not None:
            palette = colors
        else:
            if "color" in new_data_columns:
                palette = df.set_index(legend)["color"].to_dict()
            else:
                palette = []

                for i in range(len(hue_types)):
                    palette.append(plot_color_types[cmap][start_color_index + i * color_step_size + i])
    else:
        palette = colors

    # sns.set_theme(style="whitegrid")
    ax.set(ylim=y_limit)
    sns.despine()

    if is_str:
        df[x] = df[x].astype(str)

    chart = sns.lineplot(data=df, ax=ax, x=x, y=y, hue=legend, palette=palette, linewidth=line_width, **kwargs)

    if is_str:
        locator = mdates.DayLocator(interval=1)
        chart.xaxis.set_major_locator(locator)

        ax.tick_params(axis='x', rotation=x_name_rotation)
    else:
        plt.xticks(x_ticks, rotation=x_name_rotation)

    plot_end(title, x_name, y_name, output, show, close)

    return ax


def roc_prc(
    data: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    title: str = None,
    width: float = 5.5,
    height: float = 2.5,
    line_width: float = 1,
    frame_alpha: float = 0.5,
    element_title: str = 'Elements',
    legend_font_size: float = 5,
    marker_size: float = 3,
    marker_count: Union[int, float] = 10,
    method_colors: Optional[dict] = None,
    element_styles: Optional[dict] = None,
    output: Optional[path] = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Any, dict[Tuple[str, str], dict[str, float]]]:
    """
    Plot ROC and PRC curves for multiple methods and elements.

    This function creates side-by-side ROC (Receiver Operating Characteristic) and 
    PRC (Precision-Recall Curve) plots to compare the performance of different 
    methods across multiple elements. Each method is represented by a distinct color, 
    and each element is represented by a distinct linestyle and marker.

    Parameters
    ----------
    data : dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]
        Nested dictionary containing ROC and PRC data. Structure: 
        {element: {method: {"roc": {"fpr": array, "tpr": array}, 
                            "prc": {"precision": array, "recall": array}}}}
    title : Optional[str], default None
        Overall title for the figure.
    width : float, default 5.5
        Width of the figure in inches.
    height : float, default 2.5
        Height of the figure in inches.
    line_width : float, default 1
        Width of the lines in the plot.
    frame_alpha : float, default 0.5
        Transparency of the legend frame.
    legend_font_size : float, default 5
        Font size for legend text.
    element_title : str, default 'Elements'
        Title for the element legend.
    marker_size : float, default 3
        Size of the markers on the curves.
    marker_count : Union[int, float], default 10
        Number of markers to display on each curve. If > 1, specifies exact count; 
        if <= 1, specifies fraction of total points.
    method_colors : Optional[dict], default None
        Dictionary mapping method names to colors. If None, uses default color palette.
    element_styles : Optional[dict], default None
        Dictionary mapping element names to (linestyle, marker) tuples. If None, 
        uses default styles.
    output : Optional[path], default None
        File path to save the figure. If None, figure is not saved.
    show : bool, default False
        Whether to display the plot.
    close : bool, default True
        Whether to close the figure after display.
    **kwargs : Any
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    tuple[Any, dict[Tuple[str, str], dict[str, float]]]
        A tuple containing:
        - axes: Array of matplotlib Axes objects (ROC and PRC subplots)
        - results: Dictionary mapping (element, method) tuples to their auPRC and auROC scores
    """

    default_colors = plot_color_types["set"]
    default_linestyles = ['-', '--', '-.', ':']
    default_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    # Extract elements and methods from the outer dictionary
    elements = list(data.keys())
    methods = set()

    for t in elements:
        methods.update(data[t].keys())

    methods = sorted(methods)

    if method_colors is None:
        method_colors = {m: default_colors[i % len(default_colors)] for i, m in enumerate(methods)}

    if element_styles is None:
        element_styles = {
            e: (default_linestyles[i % len(default_linestyles)], default_markers[i % len(default_markers)])
            for i, e in enumerate(elements)
        }

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    ax_roc, ax_prc = axes[0], axes[1]

    # Draw curves
    for element in elements:
        # Current element style and marker
        ls, marker = element_styles[element]

        for method in methods:
            if method not in data[element]:
                continue

            # Method color
            color = method_colors[method]

            score = data[element][method]
            precision = np.asarray(score["prc"]["precision"])
            recall = np.asarray(score["prc"]["recall"])
            fpr = np.asarray(score["roc"]["fpr"])
            tpr = np.asarray(score["roc"]["tpr"])

            # Calculate AUC
            order = np.argsort(recall)
            auprc = float(np.trapz(precision[order], recall[order]))
            auroc = float(np.trapz(tpr, fpr))
            results[(element, method)] = {"auPRC": auprc, "auROC": auroc}

            # ROC curve
            ax_roc.step(fpr, tpr, where='post', color=color, linestyle=ls, linewidth=line_width, **kwargs)
            n_roc = len(fpr)
            num_markers = int(marker_count) if marker_count > 1 else max(1, int(n_roc * marker_count))
            idx_roc = np.linspace(0, n_roc - 1, num_markers).astype(int)
            ax_roc.plot(
                fpr[idx_roc],
                tpr[idx_roc],
                color=color,
                marker=marker,
                linestyle='None',
                markersize=marker_size
            )

            # PRC curve
            ax_prc.plot(recall, precision, color=color, linestyle=ls, linewidth=line_width, **kwargs)
            n_prc = len(recall)
            num_markers = int(marker_count) if marker_count > 1 else max(1, int(n_prc * marker_count))
            idx_prc = np.linspace(0, n_prc - 1, num_markers).astype(int)
            ax_prc.plot(
                recall[idx_prc],
                precision[idx_prc],
                color=color,
                marker=marker,
                linestyle='None',
                markersize=marker_size
            )

    # Calculate average AUC and build legend
    # 1. Collect AUC for each method
    method_avg = {m: {"prc": [], "roc": []} for m in methods}

    for (element, method), auc in results.items():
        method_avg[method]["prc"].append(auc["auPRC"])
        method_avg[method]["roc"].append(auc["auROC"])

    # 2. Generate method legend (with average AUC)
    method_legend = [
        Line2D(
            [0], [0], color=method_colors[m], linewidth=3,
            label=f"{m} (auPRC={np.mean(method_avg[m]['prc']):.3f}, auROC={np.mean(method_avg[m]['roc']):.3f})"
        )
        for m in methods
    ]

    # 3. Element legend
    element_legend = [
        Line2D(
            [0], [0], color='black', linestyle=element_styles[e][0],
            marker=element_styles[e][1], markersize=3, linewidth=2, label=e
        ) for e in elements
    ]

    # ROC subplot
    ax_roc.plot([0, 1], [0, 1], '--', color='gray', linewidth=line_width, alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (TPR)')
    ax_roc.set_title('ROC Comparison')
    ax_roc.set_xlim(-0.03, 1.03)
    ax_roc.set_ylim(-0.03, 1.03)
    ax_roc.grid(True, alpha=0.1)
    ax_roc.legend(
        handles=method_legend,
        loc='lower right',
        title='Methods',
        framealpha=frame_alpha,
        fontsize=legend_font_size
    )

    # PRC subplot
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title('PRC Comparison')
    ax_prc.set_xlim(-0.03, 1.03)
    ax_prc.set_ylim(-0.03, 1.03)
    ax_prc.grid(True, alpha=0.1)
    ax_prc.legend(
        handles=element_legend,
        loc='lower left',
        title=element_title,
        framealpha=frame_alpha,
        fontsize=legend_font_size
    )

    plt.tight_layout()

    plot_end(title, None, None, output, show, close)

    return axes, results

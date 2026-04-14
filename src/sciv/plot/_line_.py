# -*- coding: UTF-8 -*-

from typing import Optional, Tuple, Union, Any

from pandas import DataFrame
from anndata import AnnData
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from .. import util as ul
from ..preprocessing import adata_map_df
from ..util import path, plot_color_types, collection, plot_end, plot_start

__name__: str = "plot_line"


def base_line(
    data: Union[AnnData, DataFrame],
    x: str,
    y: str,
    layer: Optional[str] = None,
    width: float = 2,
    height: float = 2,
    bottom: float = 0,
    title: Optional[str] = None,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    label: Optional[str] = None,
    legend: Optional[str] = None,
    legend_list: list = None,
    start_color_index: int = 0,
    color_step_size: int = 0,
    color_type: str = "set",
    colors: list = None,
    line_width: float = 1.5,
    x_name_rotation: float = 65,
    x_ticks: Optional[Union[int, collection]] = None,
    y_limit: Tuple[float, float] = (0, 1),
    output: Optional[path] = None,
    is_str: bool = True,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
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
    width : float, default 2
        Figure width in inches.
    height : float, default 2
        Figure height in inches.
    bottom : float, default 0
        Bottom margin adjustment for the plot.
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
    color_type : str, default "set"
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

    Returns
    -------
    None
        The function displays and/or saves the plot but does not return any value.
    """
    fig, ax = plot_start(width, height, bottom, output, show)

    new_data = data.copy()

    if isinstance(new_data, AnnData):

        if label is not None and legend_list is not None:

            index_list = []
            label_list = list(new_data.var[label])

            for lab in range(len(label_list)):

                if legend_list.count(label_list[lab]) > 0:
                    index_list.append(lab)

            if legend_list is not None:
                new_data = new_data[:, index_list]

        # judge layers
        if layer is not None:

            if layer not in list(new_data.layers):
                ul.log(__name__).error("The value of the `layer` parameter must be one of the keys in `adata.layers`.")
                raise ValueError("The value of the `layer` parameter must be one of the keys in `adata.layers`.")

            new_data.X = new_data.layers[layer]

        # DataFrame
        ul.log(__name__).info(f"to DataFrame")
        df: DataFrame = adata_map_df(new_data, column="value")

    elif isinstance(new_data, DataFrame):

        if label is not None and legend_list is not None:
            df: DataFrame = new_data[new_data[label].isin(legend_list)].copy()
        else:
            df: DataFrame = new_data.copy()

    else:
        ul.log(__name__).error(f"The `data` parameter only support `AnnData` and `DataFrame` class types.")
        raise ValueError(f"The `data` parameter only support `AnnData` and `DataFrame` class types.")

    if legend is None and label is not None:
        legend = "category"

    if label is not None:

        df[legend] = df[label].copy()

        hue_types = df[legend].unique().tolist()

        new_data_columns = list(df.columns)

        # noinspection DuplicatedCode
        if colors is not None:
            palette = colors
        else:
            if "color" in new_data_columns:
                palette = df["color"]
            else:
                palette = []

                for i in range(len(hue_types)):
                    palette.append(plot_color_types[color_type][start_color_index + i * color_step_size + i])
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

    plot_end(fig, title, x_name, y_name, output, show, close)

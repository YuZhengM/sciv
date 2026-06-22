# -*- coding: UTF-8 -*-

from typing import Tuple, Union, Any, Literal

from matplotlib.figure import Figure
from pandas import DataFrame
import seaborn as sns

from .. import util as ul
from ..util import path, plot_end, plot_start

__name__: str = "plot_box"

log = ul.log(__name__, "ERROR")


def box(
    df: DataFrame,
    x: str = "clusters",
    y: str = "value",
    x_name: str = None,
    y_name: str = "value",
    palette: Union[Tuple, list] = None,
    line_width: float = 0.3,
    marker_size: float = 0.2,
    rotation: float = 65,
    orient: Literal["v", "h"] = "v",
    title: str = None,
    whis: float = 1.5,
    show_fliers: bool = True,
    is_sort: bool = True,
    order_names: list = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Figure, Any]:
    """
    Create a box plot with customizable styling options.

    Parameters
    ----------
    df : DataFrame
        Input data containing the values to plot.
    x : str, default "clusters"
        Column name for the x-axis categorical variable.
    y : str, default "value"
        Column name for the y-axis numerical variable.
    x_name : str, optional
        Custom label for the x-axis. If None, uses the x column name.
    y_name : str, default "value"
        Custom label for the y-axis.
    palette : Union[Tuple, list], optional
        Color palette for the boxes. If None and "color" column exists, uses that.
    line_width : float, default 0.3
        Width of lines in the plot (box edges, whiskers, etc.).
    marker_size : float, default 0.2
        Size of outlier markers.
    rotation : float, default 65
        Rotation angle for x-axis tick labels in degrees.
    orient : str, optional
        Orientation of the plot ("v" for vertical, "h" for horizontal).
    title : str, optional
        Title of the plot.
    whis : float, default 1.5
        Proportion of the IQR past the low and high quartiles to extend the whiskers.
    show_fliers : bool, default True
        Whether to display outlier points beyond the whiskers.
    is_sort : bool, default True
        Whether to sort boxes by median value in descending order.
    order_names : list, optional
        Custom order for x-axis categories. Only used if is_sort is False.
    output : path, optional
        File path to save the plot. If None, plot is not saved.
    show : bool, default True
        Whether to display the plot.
    close : bool, default False
        Whether to close the figure after displaying.
    **kwargs : Any
        Additional keyword arguments passed to seaborn.boxplot.
    """

    # judge
    df_columns = list(df.columns)

    if y not in df_columns:
        log.error(f"The `y` ({y}) parameter must be in the `df` parameter data column name ({df_columns})")
        raise ValueError(f"The `y` ({y}) parameter must be in the `df` parameter data column name ({df_columns})")

    fig, ax = plot_start()

    group_columns = [x]

    new_df: DataFrame = df.groupby(group_columns, as_index=False)[y].median()

    if "color" in df_columns:
        new_df_color: DataFrame = df.groupby(group_columns, as_index=False)["color"].first()
        new_df = new_df.merge(new_df_color, how="left", on=x)

    colors: list = []

    # sort
    if is_sort:
        new_df.sort_values([y], ascending=False, inplace=True)
        y_names: Union[list, None] = list(new_df[x])

        if "color" in df_columns:
            colors = list(new_df["color"])

    else:
        new_df.index = new_df[x]

        if order_names is not None:
            y_names: list = order_names

            if "color" in df_columns:

                for i in order_names:

                    for j, c in zip(new_df[x], new_df["color"]):

                        if i == j:
                            colors.append(c)
                            break

        else:
            y_names = new_df[x]

            if "color" in df_columns:
                colors = list(new_df["color"])

    props = {'linestyle': '-', 'linewidth': line_width}

    if orient == 'h':
        x_col, y_col = y, x
    else:
        x_col, y_col = x, y

    # scatter
    sns.boxplot(
        data=df,
        x=x_col,
        y=y_col,
        order=y_names,
        showfliers=show_fliers,
        fliersize=marker_size,
        whis=whis,
        ax=ax,
        flierprops={'marker': 'o', 'markersize': marker_size},
        boxprops=props,
        whiskerprops=props,
        medianprops=props,
        palette=palette if palette is not None else (colors if "color" in df_columns else None),
        **kwargs
    )

    lines = ax.lines

    for line in lines:
        line.set_linewidth(line_width)

    # set coordinate
    if orient == 'v':
        ax.set_xticks(range(len(y_names)))
        ax.set_xticklabels(labels=y_names, rotation=rotation)
        ax.yaxis.grid(True, linestyle='--', linewidth=line_width)
    else:
        ax.set_yticks(range(len(y_names)))
        ax.set_yticklabels(labels=y_names)
        ax.xaxis.grid(True, linestyle='--', linewidth=line_width)

    ax.spines['top'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['right'].set_linewidth(line_width)


    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, ax

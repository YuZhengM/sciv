# -*- coding: UTF-8 -*-

import os
from typing import Tuple, Union, Literal, Any

from matplotlib.figure import Figure
from pandas import DataFrame
import seaborn as sns

from .. import util as ul
from ..util import path, plot_end, plot_start

__name__: str = "plot_violin"

_Kind = Literal["strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"]

log = ul.log(__name__, "ERROR")


def violin(
    df: DataFrame,
    value: str = "value",
    x_name: str = None,
    y_name: str = "value",
    kind: _Kind = "violin",
    groupby: str = "clusters",
    palette: Union[Tuple, list] = None,
    hue: str = None,
    rotation: float = 65,
    line_width: float = 0.5,
    title: str = None,
    split: bool = False,
    is_sort: bool = True,
    order_names: list = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Figure, Any]:
    """
    Plot violin plot.

    Parameters
    ----------
    df : DataFrame
        Input data.
    value : str, optional
        Value column.
    x_name : str, optional
        X name.
    y_name : str, optional
        Y name.
    kind : _Kind, optional
        Kind of plot.
    groupby : str, optional
        Clusters column.
    palette : Union[Tuple, list], optional
        Palette.
    hue : str, optional
        Hue column.
    rotation : float, optional
        Rotation.
    line_width : float, optional
        Line width.
    title : str, optional
        Title.
    split : bool, optional
        Whether to split.
    is_sort : bool, optional
        Whether to sort.
    order_names : list, optional
        Order names.
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
    # judge
    df_columns = list(df.columns)

    if value not in df_columns:
        log.error(f"The `value` ({value}) parameter must be in the `df` parameter data column name ({df_columns})")
        raise ValueError(
            f"The `value` ({value}) parameter must be in the `df` parameter data column name ({df_columns})"
        )

    if hue is not None and hue not in df_columns:
        log.error(f"The `hue` ({hue}) parameter must be in the `df` parameter data column name ({df_columns})")
        raise ValueError(f"The `hue` ({hue}) parameter must be in the `df` parameter data column name ({df_columns})")

    fig, ax = plot_start()

    group_columns = [groupby]

    new_df: DataFrame = df.groupby(group_columns, as_index=False)[value].median()

    if "color" in df_columns:
        new_df_color: DataFrame = df.groupby(group_columns, as_index=False)["color"].first()
        new_df = new_df.merge(new_df_color, how="left", on=groupby)

    colors: list = []

    # sort
    if is_sort:
        new_df.sort_values([value], ascending=False, inplace=True)
        y_names: Union[list, None] = list(new_df[groupby])

        if "color" in df_columns:
            colors = list(new_df["color"])

    else:
        new_df.index = new_df[groupby]

        if order_names is not None:
            y_names: list = order_names

            if "color" in df_columns:

                for i in order_names:

                    for j, c in zip(new_df[groupby], new_df["color"]):

                        if i == j:
                            colors.append(c)
                            break

        else:
            y_names = list(new_df[groupby])

            if "color" in df_columns:
                colors = list(new_df["color"])

    # scatter
    g = sns.catplot(
        data=df,
        x=groupby,
        y=value,
        kind=kind,
        hue=hue,
        order=y_names,
        split=split,
        linewidth=line_width,
        palette=palette if palette is not None else (colors if "color" in df_columns else None),
        **kwargs
    )

    # set coordinate
    for _ax_ in g.axes.flat:
        _ax_.spines['top'].set_linewidth(line_width)
        _ax_.spines['right'].set_linewidth(line_width)
        _ax_.spines['bottom'].set_linewidth(line_width)
        _ax_.spines['left'].set_linewidth(line_width)
        # Set the rotation angle of the x-axis labels
        _ax_.tick_params(axis='x', rotation=rotation)

    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, g

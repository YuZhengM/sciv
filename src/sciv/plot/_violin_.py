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


def violin_base(
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


def violin_trait(
    trait_df: DataFrame,
    trait_name: Union[str, list] = "All",
    trait_column_name: str = "id",
    value: str = "value",
    groupby: str = "clusters",
    kind: _Kind = "violin",
    x_name: str = None,
    y_name: str = "value",
    palette: Tuple = None,
    rotation: float = 65,
    line_width: float = 0.1,
    split: bool = False,
    is_sort: bool = True,
    order_names: list = None,
    title: str = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Plot violin plot for trait data.

    This function creates violin plots (or other categorical plots) for trait data,
    allowing visualization of trait distributions across different clusters.

    Parameters
    ----------
    trait_df : DataFrame
        Input trait data containing trait information and values.
    trait_name : Union[str, list], optional
        Name(s) of the trait(s) to plot. Use "All" to plot all traits.
    trait_column_name : str, optional
        Column name in trait_df that contains trait identifiers.
    value : str, optional
        Column name containing the values to plot.
    groupby : str, optional
        Column name containing cluster assignments.
    kind : _Kind, optional
        Type of categorical plot to create (e.g., "violin", "box", "strip").
    x_name : str, optional
        Label for the x-axis.
    y_name : str, optional
        Label for the y-axis.
    palette : Tuple, optional
        Color palette for the plot.
    rotation : float, optional
        Rotation angle for x-axis labels in degrees.
    line_width : float, optional
        Width of the plot lines.
    split : bool, optional
        Whether to split the violin plot when using hue.
    is_sort : bool, optional
        Whether to sort clusters by median value.
    order_names : list, optional
        Custom order for cluster names.
    title : str, optional
        Title prefix for the plot.
    output : path, optional
        Directory path to save the output files.
    show : bool, optional
        Whether to display the plot.
    close : bool, optional
        Whether to close the figure after saving.
    kwargs : Any, optional
        Additional keyword arguments passed to violin_base.

    Returns
    -------
    None
    """
    data: DataFrame = trait_df.copy()

    def trait_plot(_trait_: Union[str, list], _cell_df_: DataFrame) -> None:
        """
        Plot trait violin plot.
        
        Parameters
        ----------
        _trait_ : Union[str, list]
            Trait name.
        _cell_df_ : DataFrame
            Cell data.
            
        Returns
        -------
        None
        """
        log.info("Plotting box {}".format(_trait_))
        # Get gene score
        _filename_: str = _trait_
        trait_score = _cell_df_[_cell_df_[trait_column_name] == _trait_]
        # Sort gene scores from small to large
        violin_base(
            df=trait_score,
            value=value,
            x_name=x_name,
            y_name=y_name,
            palette=palette,
            split=split,
            is_sort=is_sort,
            rotation=rotation,
            order_names=order_names,
            kind=kind,
            hue=trait_column_name,
            line_width=line_width,
            groupby=groupby,
            title=f"{title} {_filename_}" if title is not None else title,
            output=os.path.join(output, f"cell_{_filename_}_score_cat_{kind}.pdf") if output is not None else None,
            show=show,
            close=close,
            **kwargs
        )

    # noinspection DuplicatedCode
    trait_list = list(set(data[trait_column_name]))
    # Validate trait
    if trait_name != "All":
        if isinstance(trait_name, str):
            if trait_name not in trait_list:
                log.error(f"The {trait_name} trait/disease is not in the trait/disease list {trait_list}.")
                raise ValueError(f"The {trait_name} trait/disease is not in the trait/disease list {trait_list}.")
        else:
            for tn in trait_name:
                if tn not in trait_list:
                    log.error(f"The {tn} trait/disease is not in the trait/disease list {trait_list}.")
                    raise ValueError(f"The {tn} trait/disease is not in the trait/disease list {trait_list}.")

    # Plot
    if trait_name == "All":
        for trait in trait_list:
            trait_plot(trait, trait_df)
    else:
        trait_plot(trait_name, trait_df)

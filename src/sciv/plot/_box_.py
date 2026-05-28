# -*- coding: UTF-8 -*-

import os
from typing import Tuple, Union, Any

from pandas import DataFrame
import seaborn as sns

from .. import util as ul
from ..util import path, plot_end, plot_start

__name__: str = "plot_box"

log = ul.log(__name__, "ERROR")

def box_base(
    df: DataFrame,
    x: str = "clusters",
    y: str = "value",
    x_name: str = None,
    y_name: str = "value",
    palette: Union[Tuple, list] = None,
    line_width: float = 0.3,
    marker_size: float = 0.2,
    rotation: float = 65,
    orient: str = None,
    title: str = None,
    whis: float = 1.5,
    show_fliers: bool = True,
    is_sort: bool = True,
    order_names: list = None,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
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

    fig, ax = plot_start(output, show)

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

    # scatter
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        order=y_names,
        showfliers=show_fliers,
        fliersize=marker_size,
        orient=orient,
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
    ax.set_xticks(range(len(y_names)))
    ax.set_xticklabels(labels=y_names, rotation=rotation)
    ax.spines['top'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['right'].set_linewidth(line_width)

    ax.yaxis.grid(True, linestyle='-', linewidth=line_width)

    plot_end(fig, title, x_name, y_name, output, show, close)


def box_trait(
    trait_df: DataFrame,
    trait_name: str = "All",
    trait_column_name: str = "id",
    value: str = "value",
    groupby: str = "clusters",
    x_name: str = None,
    y_name: str = "value",
    palette: Union[Tuple, list] = None,
    orient: str = None,
    line_width: float = 0.1,
    marker_size: float = 0.5,
    rotation: float = 65,
    whis: float = 1.5,
    show_fliers: bool = True,
    is_sort: bool = True,
    order_names: list = None,
    title: str = None,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
    """
    Create box plots for trait/disease data across different clusters.

    This function generates box plots for each trait or a specific trait from the input dataframe.
    It filters data by trait and creates individual box plots using the box_base function.

    Parameters
    ----------
    trait_df : DataFrame
        Input data containing trait/disease information and values to plot.
    trait_name : str, default "All"
        Name of the trait/disease to plot. Use "All" to plot all traits.
    trait_column_name : str, default "id"
        Column name in trait_df that contains trait/disease identifiers.
    value : str, default "value"
        Column name for the numerical values to be plotted on y-axis.
    groupby : str, default "clusters"
        Column name for the cluster categories to be plotted on x-axis.
    x_name : str, optional
        Custom label for the x-axis. If None, uses the clusters column name.
    y_name : str, default "value"
        Custom label for the y-axis.
    palette : Union[Tuple, list], optional
        Color palette for the boxes.
    orient : str, optional
        Orientation of the plot ("v" for vertical, "h" for horizontal).
    line_width : float, default 0.1
        Width of lines in the plot.
    marker_size : float, default 0.5
        Size of outlier markers.
    rotation : float, default 65
        Rotation angle for x-axis tick labels in degrees.
    whis : float, default 1.5
        Proportion of the IQR to extend the whiskers.
    show_fliers : bool, default True
        Whether to display outlier points beyond the whiskers.
    is_sort : bool, default True
        Whether to sort boxes by median value.
    order_names : list, optional
        Custom order for x-axis categories.
    title : str, optional
        Base title for the plots. Trait name will be appended.
    output : path, optional
        Directory path to save the plots. If None, plots are not saved.
    show : bool, default True
        Whether to display the plots.
    close : bool, default False
        Whether to close the figure after displaying.
    **kwargs : Any
        Additional keyword arguments passed to box_base function.
    """

    data: DataFrame = trait_df.copy()

    def trait_plot(trait_: str, atac_cell_df_: DataFrame) -> None:
        """
        Generate a box plot for a specific trait.

        Parameters
        ----------
        trait_ : str
            The name of the trait/disease to plot.
        atac_cell_df_ : DataFrame
            The input dataframe containing trait data and values.
        """
        log.info("Plotting box {}".format(trait_))
        # get gene score
        trait_score = atac_cell_df_[atac_cell_df_[trait_column_name] == trait_]
        # Sort gene scores from small to large
        box_base(
            df=trait_score,
            x=groupby,
            y=value,
            x_name=x_name,
            y_name=y_name,
            palette=palette,
            rotation=rotation,
            is_sort=is_sort,
            whis=whis,
            order_names=order_names,
            line_width=line_width,
            show_fliers=show_fliers,
            marker_size=marker_size,
            orient=orient,
            title=f"{title} {trait_}" if title is not None else title,
            output=os.path.join(output, f"cell_{trait_}_score_box.pdf") if output is not None else None,
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

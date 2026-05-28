# -*- coding: UTF-8 -*-

import os
from typing import Tuple, Union, Optional, Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from pandas import DataFrame

import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

from ._util_ import complete_ratio
from .. import util as ul
from ..util import path, collection, plot_color_types, plot_start, plot_end

__name__: str = "plot_bar"

log = ul.log(__name__, "ERROR")

def bar(
    ax_x: collection,
    ax_y: collection,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    color: str = "#70b5de",
    text_color: str = "#000205",
    text_left_move: float = 0.1,
    direction: Literal['vertical', 'horizontal'] = "vertical",
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
    """
    Create a simple bar chart with optional value labels.

    This function generates a bar plot (vertical or horizontal) with customizable
    appearance and automatically adds numerical value labels on each bar.

    Parameters
    ----------
    ax_x : collection
        Categories or labels for the x-axis (or y-axis if horizontal).
    ax_y : collection
        Numerical values for the bar heights (or widths if horizontal).
    x_name : str, optional
        Label for the x-axis. Default is None.
    y_name : str, optional
        Label for the y-axis. Default is None.
    title : str, optional
        Title of the plot. Default is None.
    color : str, default "#70b5de"
        Color of the bars.
    text_color : str, default "#000205"
        Color of the value labels on bars.
    text_left_move : float, default 0.1
        Horizontal adjustment for text position on bars.
    direction : Literal['vertical', 'horizontal'], default "vertical"
        Orientation of the bars.
    output : path, optional
        File path to save the figure. Default is None.
    show : bool, default True
        Whether to display the plot.
    close : bool, default False
        Whether to close the figure after saving.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib's bar/barh function.

    Returns
    -------
    None
        The function displays and/or saves the plot but does not return any value.
    """
    fig, ax = plot_start(output, show)

    ax_x = np.array(ax_x).astype(str)

    if direction == 'vertical':
        ax.bar(ax_x, ax_y, color=color, **kwargs)
    elif direction == 'horizontal':
        ax.barh(ax_x, ax_y, color=color, **kwargs)
    else:
        log.error("The `direction` must be 'vertical' or 'horizontal'.")
        raise ValueError("The `direction` must be 'vertical' or 'horizontal'.")

    ax.set_xticklabels(labels=list(ax_x), rotation=65)

    # Draw numerical values
    for i, v in enumerate(list(ax_y)):
        plt.text(
            x=i - text_left_move,
            y=0.03 if v < 0.03 else v / 2,
            s=str(round(v, 3)),
            rotation=90,
            color=text_color
        )

    plot_end(fig, title, x_name, y_name, output, show, close)


def two_bar(
    ax_x: collection,
    ax_y: Tuple,
    x_name: str = None,
    y_name: str = None,
    legend: Tuple = ("1", "2"),
    color: Tuple = ("#2e6fb7", "#f7f7f7"),
    text_color: str = "#000205",
    rotation: float = 65,
    text_left_move: float = 0.15,
    y_limit: Tuple[float, float] = (0, 1),
    title: str = None,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
):
    """
    Create a stacked bar chart with two categories.

    This function generates a stacked bar plot where two sets of values are displayed
    as stacked bars. It automatically adds numerical value labels on the first bar segment
    and includes a legend for the two categories.

    Parameters
    ----------
    ax_x : collection
        Categories or labels for the x-axis.
    ax_y : Tuple
        A tuple containing two collections of numerical values for the two bar segments.
        The second segment will be stacked on top of the first.
    x_name : str, optional
        Label for the x-axis. Default is None.
    y_name : str, optional
        Label for the y-axis. Default is None.
    legend : Tuple, default ("1", "2")
        Labels for the legend corresponding to the two bar segments.
    color : Tuple, default ("#2e6fb7", "#f7f7f7")
        Colors for the two bar segments (first segment, second segment).
    text_color : str, default "#000205"
        Color of the value labels on bars.
    rotation : float, default 65
        Rotation angle for x-axis tick labels in degrees.
    text_left_move : float, default 0.15
        Horizontal adjustment for text position on bars.
    y_limit : Tuple[float, float], default (0, 1)
        The y-axis limits for the plot.
    title : str, optional
        Title of the plot. Default is None.
    output : path, optional
        File path to save the figure. Default is None.
    show : bool, default True
        Whether to display the plot.
    close : bool, default False
        Whether to close the figure after saving.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib's bar function.

    Returns
    -------
    None
        The function displays and/or saves the plot but does not return any value.
    """
    fig, ax = plot_start(output, show)

    ax_x = np.array(ax_x).astype(str)
    ax.bar(ax_x, ax_y[0], label=legend[0], color=color[0], **kwargs)
    ax.bar(ax_x, ax_y[1], bottom=ax_y[0], label=legend[1], color=color[1], **kwargs)

    ax.legend()

    ax.set_ylim(y_limit)

    ax.set_xticks(range(len(ax_x)))
    ax.set_xticklabels(labels=list(ax_x), rotation=rotation)

    # Draw numerical values
    for i, v in enumerate(list(ax_y[0])):
        plt.text(
            x=i - text_left_move,
            y=0.03 if v < 0.03 else v / 2,
            s=str(round(v, 3)),
            rotation=90,
            color=text_color
        )

    for spine in ["top", "left", "right", "bottom"]:
        ax.spines[spine].set_linewidth(1)

    ax.spines['bottom'].set_linewidth(1)
    ax.grid(axis='y', ls='--', c='gray')
    ax.set_axisbelow(True)

    plot_end(fig, title, x_name, y_name, output, show, close)


def class_bar(
    df: DataFrame,
    value: str = "rate",
    by: str = "value",
    groupby: str = "clusters",
    color: Tuple = ("#2e6fb7", "#f7f7f7"),
    x_name: str = "Cell type",
    y_name: str = "Enrichment ratio",
    legend: Tuple = ("Enrichment", "Conservative"),
    text_color: str = "#000205",
    groupby_sort: Optional[list] = None,
    rotation: float = 65,
    title: str = None,
    text_left_move: float = 0.15,
    y_limit: Tuple[float, float] = (0, 1),
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
):
    """
    Create a stacked bar chart for enrichment analysis with two categories.

    This function filters a DataFrame by a binary column, sorts the data by clusters,
    and generates a stacked bar plot using the two_bar function. It is typically used
    to visualize enrichment ratios where one category represents enriched values and
    the other represents conservative values.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the data to plot.
    value : str, default "rate"
        Column name containing the numerical values to plot.
    by : str, default "value"
        Column name used to filter the DataFrame into two categories (typically binary: 0 and 1).
    groupby : str, default "clusters"
        Column name containing the cluster labels or categories.
    color : Tuple, default ("#2e6fb7", "#f7f7f7")
        Colors for the two bar segments (enrichment color, conservative color).
    x_name : str, default "Cell type"
        Label for the x-axis.
    y_name : str, default "Enrichment ratio"
        Label for the y-axis.
    legend : Tuple, default ("Enrichment", "Conservative")
        Labels for the legend corresponding to the two bar segments.
    text_color : str, default "#000205"
        Color of the value labels on bars.
    groupby_sort : Optional[list], default None
        Custom order for clusters. If provided, clusters will be sorted according to this list.
        If None, clusters will be sorted by value in descending order.
    rotation : float, default 65
        Rotation angle for x-axis tick labels in degrees.
    title : str, optional
        Title of the plot. Default is None.
    text_left_move : float, default 0.15
        Horizontal adjustment for text position on bars.
    y_limit : Tuple[float, float], default (0, 1)
        The y-axis limits for the plot.
    output : path, optional
        File path to save the figure. Default is None.
    show : bool, default True
        Whether to display the plot.
    close : bool, default False
        Whether to close the figure after saving.
    **kwargs : Any
        Additional keyword arguments passed to the two_bar function.

    Returns
    -------
    None
        The function displays and/or saves the plot but does not return any value.
    """
    df1 = df[df[by] == 1]
    df2 = df[df[by] == 0]

    # Sort
    if groupby_sort is not None:
        df1[groupby] = pd.Categorical(df1[groupby], categories=groupby_sort, ordered=True)
        df1 = df1.sort_values(by=groupby)
        df2[groupby] = pd.Categorical(df2[groupby], categories=groupby_sort, ordered=True)
        df2 = df2.sort_values(by=groupby)
        ax_x = groupby_sort
    else:
        df1 = df1.sort_values([value], ascending=False)
        df2 = df2.sort_values([value])
        ax_x = df1[groupby]

    ax_y = (df1[value], df2[value])

    two_bar(
        ax_x=ax_x,
        ax_y=ax_y,
        x_name=x_name,
        y_name=y_name,
        legend=legend,
        color=color,
        text_color=text_color,
        rotation=rotation,
        text_left_move=text_left_move,
        y_limit=y_limit,
        title=title,
        output=output,
        show=show,
        close=close,
        **kwargs
    )


def bar_trait(
    trait_df: DataFrame,
    trait_name: str = "All",
    trait_column_name: str = "id",
    value: str = "rate",
    groupby: str = "clusters",
    x_name: str = "Cell type",
    y_name: str = "Enrichment ratio",
    color: Tuple = ("#2e6fb7", "#f7f7f7"),
    legend: Tuple = ("Enrichment", "Conservative"),
    text_color: str = "#000205",
    groupby_sort: Optional[list] = None,
    rotation: float = 65,
    title: str = None,
    text_left_move: float = 0.15,
    y_limit: Tuple[float, float] = (0, 1),
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
):
    """
    Create stacked bar charts for multiple traits or a specific trait.

    This function generates enrichment bar plots for traits (e.g., diseases, gene sets)
    in the input DataFrame. It can plot all traits or a specific trait based on the
    trait_name parameter. Each trait's enrichment data is visualized using the class_bar
    function, with results saved to individual files.

    Parameters
    ----------
    trait_df : DataFrame
        Input DataFrame containing trait enrichment data. Must include columns for
        trait identifiers, cluster labels, and enrichment values.
    trait_name : str, default "All"
        The specific trait to plot. If "All", plots bar charts for all unique traits
        in the trait_column_name column.
    trait_column_name : str, default "id"
        Column name in trait_df that contains trait identifiers.
    value : str, default "rate"
        Column name containing the numerical enrichment values to plot.
    groupby : str, default "clusters"
        Column name containing cluster or cell type labels.
    x_name : str, default "Cell type"
        Label for the x-axis.
    y_name : str, default "Enrichment ratio"
        Label for the y-axis.
    color : Tuple, default ("#2e6fb7", "#f7f7f7")
        Colors for the two bar segments (enrichment color, conservative color).
    legend : Tuple, default ("Enrichment", "Conservative")
        Labels for the legend corresponding to the two bar segments.
    text_color : str, default "#000205"
        Color of the value labels on bars.
    groupby_sort : Optional[list], default None
        Custom order for clusters. If provided, clusters will be sorted according
        to this list. If None, clusters are sorted by enrichment value.
    rotation : float, default 65
        Rotation angle for x-axis tick labels in degrees.
    title : str, optional
        Base title of the plot. The trait name will be appended to this title.
        Default is None.
    text_left_move : float, default 0.15
        Horizontal adjustment for text position on bars.
    y_limit : Tuple[float, float], default (0, 1)
        The y-axis limits for the plot.
    output : path, optional
        Directory path to save the figures. If provided, each trait's plot will be
        saved as a PDF file in this directory. Default is None.
    show : bool, default True
        Whether to display the plot.
    close : bool, default False
        Whether to close the figure after saving.
    **kwargs : Any
        Additional keyword arguments passed to the class_bar function.

    Returns
    -------
    None
        The function displays and/or saves the plots but does not return any value.
    """

    def trait_plot(trait_: str, cell_df_: DataFrame) -> None:
        """
        show plot
        :param trait_: trait name
        :param cell_df_:
        :return: None
        """
        log.info("Plotting bar {}".format(trait_))
        # get gene score
        trait_score = cell_df_[cell_df_[trait_column_name] == trait_]
        # Sort gene scores from small to large
        class_bar(
            df=trait_score,
            value=value,
            groupby=groupby,
            title=f"{title} {trait_}" if title is not None else title,
            color=color,
            legend=legend,
            x_name=x_name,
            y_name=y_name,
            groupby_sort=groupby_sort,
            rotation=rotation,
            text_left_move=text_left_move,
            y_limit=y_limit,
            text_color=text_color,
            output=os.path.join(output, f"cell_{trait_}_enrichment_bar.pdf") if output is not None else None,
            show=show,
            close=close,
            **kwargs
        )

    trait_list = list(set(trait_df[trait_column_name]))

    # judge trait
    if trait_name != "All" and trait_name not in trait_list:
        log.error(
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


def bar_significance(
    df: DataFrame,
    x: str,
    y: str,
    hue: str,
    x_name: str = None,
    y_name: str = None,
    anchor: str = None,
    legend: str = None,
    legend_list: list = None,
    hue_order: list = None,
    legend_gap: float = 1.15,
    line_width: float = 0.5,
    capsize: float = 0.1,
    errcolor: str = "k",
    start_color_index: int = 0,
    color_step_size: int = 0,
    color_type: str = "set",
    test: str = "t-test_ind",
    ci: Union[str, float] = "sd",
    x_rotation: float = 0,
    x_deviation: float = 0.02,
    y_deviation: float = 0.02,
    y_limit: Tuple[float, float] = (0, 1),
    anno: bool = False,
    anno_fontsize: float = 7,
    line_height: float = 0.01,
    line_offset: float = 0.01,
    colors: Union[list, dict] = None,
    title: str = None,
    output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
    """
    Create a bar chart with statistical significance annotations relative to an anchor group.

    This function generates a grouped bar plot with error bars and performs pairwise
    statistical significance testing between an anchor group and other groups within
    each category. It supports custom color palettes, legend positioning, and various
    statistical tests.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the data to plot.
    x : str
        Column name for x-axis categories.
    y : str
        Column name for y-axis values.
    hue : str
        Column name for grouping bars by color.
    x_name : str, optional
        Label for x-axis. Default is None.
    y_name : str, optional
        Label for y-axis. Default is None.
    anchor : str, optional
        Reference group name for pairwise significance testing. If provided, statistical
        comparisons will be made between this group and all other groups within each x category.
    legend : str, optional
        Legend title. Default is "category".
    legend_list : list, optional
        Subset of hue values to include in the plot. If provided, only these values
        will be plotted. Default is None.
    hue_order : list, optional
        Order of hue categories for plotting and legend. Default is None.
    legend_gap : float, default 1.15
        Vertical gap between plot and legend, specified as a ratio of the y-axis height.
    line_width : float, default 0.5
        Width of error bars and significance annotation lines.
    capsize : float, default 0.1
        Width of the error bar caps.
    errcolor : str, default "k"
        Color of the error bars.
    start_color_index : int, default 0
        Starting index in the color palette for the first hue category.
    color_step_size : int, default 0
        Step size when cycling through the color palette for subsequent hue categories.
    color_type : str, default "set"
        Name of the seaborn color palette to use. Must be a key in plot_color_types.
    test : str, default "t-test_ind"
        Statistical test for pairwise comparisons. Options include:
        {"t-test_ind", "t-test_welch", "t-test_paired", "Mann-Whitney", "Mann-Whitney-gt",
         "Mann-Whitney-ls", "Levene", "Wilcoxon", "Kruskal", "Brunner-Munzel"}.
    ci : Union[str, float], default "sd"
        Confidence interval type or value for error bars. Can be "sd" for standard deviation
        or a float for confidence interval percentage.
    x_rotation : float, default 0
        Rotation angle for x-axis tick labels in degrees.
    x_deviation : float, default 0.02
        Horizontal offset for bar value annotations.
    y_deviation : float, default 0.02
        Vertical offset adjustment for bar value annotations.
    y_limit : Tuple[float, float], default (0, 1)
        Y-axis limits for the plot.
    anno : bool, default False
        Whether to annotate bars with their numerical values.
    anno_fontsize : float, default 7
        Font size for bar value annotations.
    line_height : float, default 0.01
        Height of significance annotation lines as a fraction of y-axis range.
    line_offset : float, default 0.01
        Vertical offset for significance annotation lines from the bar tops.
    colors : Union[list, dict], optional
        Custom color list or dictionary mapping hue values to colors. If provided,
        overrides the default color palette. Default is None.
    title : str, optional
        Title of the plot. Default is None.
    output : path, optional
        File path to save the figure. Default is None.
    show : bool, default True
        Whether to display the plot.
    close : bool, default False
        Whether to close the figure after saving.
    **kwargs : Any
        Additional keyword arguments passed to seaborn's barplot function.

    Returns
    -------
    None
        The function displays and/or saves the plot but does not return any value.
    """
    fig, ax = plot_start(output, show)

    if legend_list is not None:
        new_data: DataFrame = df[df[hue].isin(legend_list)].copy()
    else:
        new_data: DataFrame = df.copy()

    if legend is None:
        legend = "category"

    new_data.loc[:, legend] = new_data[hue].astype(str)

    new_data_columns = list(new_data.columns)

    if hue_order is not None:
        # Sort
        new_data[legend] = pd.Categorical(new_data[legend], categories=hue_order, ordered=True)
        new_data = new_data.sort_values(by=legend)

    hue_types = new_data[legend].unique().tolist()

    if colors is not None:
        if isinstance(colors, list):
            palette = colors
        elif isinstance(colors, dict):
            palette = []
            for hue_type in hue_types:
                if hue_type in colors:
                    palette.append(colors[hue_type])
                else:
                    log.warning(f"`{hue_type}` is not in `colors` ({colors})")
                    raise ValueError(f"`{hue_type}` is not in `colors` ({colors})")
        else:
            log.error(f"`colors` ({colors}) must be a list or dict")
            raise ValueError(f"`colors` ({colors}) must be a list or dict")
    else:
        if "color" in new_data_columns:
            palette = new_data["color"]
        else:
            palette = []

            for i in range(len(new_data[legend])):
                _index_ = hue_types.index(i)
                palette.append(plot_color_types[color_type][start_color_index + _index_ * color_step_size + _index_])

    palette_dict = dict(zip(new_data[legend], palette))

    # Set y-axis limits first to prevent seaborn from overriding
    ax.set_ylim(y_limit)
    # Draw barplot, note ax receives return value to keep handles
    ax = sns.barplot(
        data=new_data,
        x=x,
        y=y,
        hue=legend,
        hue_order=hue_order,
        errorbar=ci if isinstance(ci, str) else ('ci', ci),
        capsize=capsize,
        err_kws={'color': errcolor, 'linewidth': line_width},
        ax=ax,
        palette=palette_dict,
        edgecolor=errcolor,
        linewidth=line_width,
        **kwargs
    )

    if anno:
        for p in ax.patches:
            y_value = p.get_height()
            height = p.get_height() / 2 - y_deviation
            height = 0.03 if height < 0.03 else height
            x = p.get_x() + p.get_width() / 2 + x_deviation
            ax.annotate(
                f'{y_value:.2f}',
                (x, height),
                textcoords="offset points",
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=anno_fontsize
            )

    if anchor is not None:

        # Add p value
        box_pairs: list = []

        x_list = new_data[x].unique().tolist()
        class_list = new_data[legend].unique().tolist()

        if anchor not in class_list:
            log.error(f"`anchor` ({anchor}) is not in the `df[hue]` ({class_list})")
            raise ValueError(f"`anchor` ({anchor}) is not in the `df[hue]` ({class_list})")

        class_list.remove(anchor)

        for x_ele in x_list:

            for class_ele in class_list:
                box_pairs.append(((x_ele, anchor), (x_ele, class_ele)))

        annotator = Annotator(ax=ax, data=new_data, x=x, y=y, hue=legend, hue_order=hue_order, pairs=box_pairs)
        annotator.configure(test=test, text_format='star', line_height=line_height, line_offset=line_offset,
                            line_width=0.7)
        annotator.apply_and_annotate()

    ax.tick_params(which='major', direction='in', length=3, width=1.0, bottom=False)

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines['bottom'].set_linewidth(1)
    ax.grid(axis='y', ls='--', c='gray')
    ax.set_axisbelow(True)

    if x_rotation != 0:
        ax.tick_params(axis='x', rotation=x_rotation)

    plt.legend(loc='upper left', bbox_to_anchor=(0.0, legend_gap), ncol=2)

    plot_end(fig, title, x_name, y_name, output, show, close)


def rate_bar_plot(
    adata: AnnData,
    layer: str = None,
    trait_name: str = "All",
    dir_name: str = "feature",
    column: str = "value",
    groupby: str = "clusters",
    color: Tuple = ("#2e6fb7", "#f7f7f7"),
    legend: Tuple = ("Enrichment", "Conservative"),
    x_name: str = "Cell type",
    y_name: str = "Enrichment ratio",
    groupby_sort: Optional[list] = None,
    text_color: str = "#000205",
    rotation: float = 65,
    title: str = None,
    text_left_move: float = 0.15,
    y_limit: Tuple[float, float] = (0, 1),
    plot_output: path = None,
    show: bool = True,
    close: bool = False,
    **kwargs: Any
) -> None:
    """
    Generate a bar plot showing enrichment ratios for trait-cluster combinations.

    This function calculates the completion ratio using the complete_ratio function
    and visualizes the results as a bar plot. It handles directory creation for
    output files and passes appropriate parameters to the bar_trait plotting function.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing the data to be visualized.
    layer : str, optional
        Specify the layer of the matrix to be processed. If None, uses the main matrix.
    trait_name : str, default "All"
        The name of the trait being analyzed, used for filtering data.
    dir_name : str, default "feature"
        Folder name for generating and saving bar plot outputs.
    column : str, default "value"
        The column name containing the binary enrichment values.
    groupby : str, default "clusters"
        The column name in adata.obs that defines the cell clusters.
    color : Tuple, default ("#2e6fb7", "#f7f7f7")
        Color tuple for the bar plot (enrichment color, conservative color).
    legend : Tuple, default ("Enrichment", "Conservative")
        Legend labels for the two categories in the plot.
    x_name : str, default "Cell type"
        The label for the x-axis.
    y_name : str, default "Enrichment ratio"
        The label for the y-axis.
    groupby_sort : Optional[list], optional
        Custom order for clusters. If None, uses default sorting.
    text_color : str, default "#000205"
        Color for text annotations in the plot.
    rotation : float, default 65
        Rotation angle for x-axis labels in degrees.
    title : str, optional
        The title of the plot. If None, no title is displayed.
    text_left_move : float, default 0.15
        Horizontal adjustment for text position.
    y_limit : Tuple[float, float], default (0, 1)
        The y-axis limits for the plot.
    plot_output : path, optional
        Directory path for saving output files. If None, figures are not saved.
    show : bool, default True
        If True, display the figure interactively.
    close : bool, default False
        If True, close the figure after saving.
    **kwargs : Any
        Additional keyword arguments passed to bar_trait function.

    Returns
    -------
    None
        This function does not return any value. Outputs are saved to files or displayed.
    """

    if dir_name is not None:
        # create path
        new_path = os.path.join(plot_output, f"{dir_name}_{layer}") if plot_output is not None else None
        # create path
        if plot_output is not None:
            ul.file_method(__name__).makedirs(new_path)
    else:
        plot_output = None
        new_path = None

    # create data
    new_value_group = complete_ratio(adata=adata, layer=layer, column=column, groupby=groupby)

    bar_trait(
        trait_df=new_value_group,
        value="rate",
        groupby=groupby,
        title=title,
        x_name=x_name,
        y_name=y_name,
        groupby_sort=groupby_sort,
        trait_name=trait_name,
        color=color,
        legend=legend,
        text_color=text_color,
        rotation=rotation,
        text_left_move=text_left_move,
        y_limit=y_limit,
        output=new_path if plot_output is not None else None,
        show=show,
        close=close,
        **kwargs
    )

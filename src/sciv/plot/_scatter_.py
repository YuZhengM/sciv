# -*- coding: UTF-8 -*-

import os.path
from typing import Union, Tuple, Optional, Any

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pandas import DataFrame
import seaborn as sns

from .. import util as ul
from ..util import path, collection, type_50_colors, type_20_colors, chrtype, type_set_colors, plot_end, plot_start

__name__: str = "plot_pie"

log = ul.log(__name__, "ERROR")


def scatter(
    df: DataFrame,
    x: str,
    y: str,
    hue: str,
    hue_order: list = None,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    cmap: str = "Oranges",
    text_fontsize: float = 7,
    text_lw: float = 1,
    start_color_index: int = 0,
    color_step_size: int = 0,
    type_colors: collection = None,
    bar_title: str = None,
    edge_color: str = None,
    size: Union[float, collection] = 1.0,
    legend: dict = None,
    number: bool = False,
    is_text: bool = False,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Create a base scatter plot with customizable aesthetics.
    
    Parameters
    ----------
    df : DataFrame
        Input data containing x, y coordinates and optional hue values
    x : str
        Column name for x-axis values
    y : str
        Column name for y-axis values
    hue : str, optional
        Column name for color grouping
    hue_order : list, optional
        Order of hue categories for legend
    x_name : str, optional
        Label for x-axis
    y_name : str, optional
        Label for y-axis
    title : str, optional
        Plot title
    cmap : str, default "Oranges"
        Colormap for continuous coloring
    text_lw : float, default 1
    text_fontsize : float, default 7
        Font size for annotation text
    start_color_index : int, default 0
        Starting index in color palette
    color_step_size : int, default 0
        Step size for color selection
    bar_title : str, optional
        Label for colorbar when number=True
    type_colors : collection, optional
        Custom color palette
    edge_color : str, optional
        Edge color for scatter points
    size : Union[float, collection], default 1.0
        Size of scatter points
    legend : dict, optional
        Mapping to rename hue categories
    number : bool, default False
        Whether to use continuous color scale
    is_text : bool, default False
        Whether to add text annotations
    output : path, optional
        Output file path
    show : bool, default True
        Whether to display the plot
    close : bool, default False
        Whether to close the figure after saving
    **kwargs : Any
        Additional arguments passed to sns.scatterplot
    """
    fig, ax = plot_start()

    # scatter
    if number:
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            ax=ax,
            palette=cmap,
            hue=hue,
            s=size,
            legend=False,
            edgecolor=edge_color,
            **kwargs
        )

        # Create continuous color scale for numerical hue values
        norm = plt.Normalize(df[hue].min(), df[hue].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax)

        if bar_title is not None:
            cbar.set_label(bar_title, ha='center', va='top')
    else:
        # Get unique hue categories and sort them
        __hue_order__ = list(np.sort(list(set(df[hue]))))

        # Select appropriate default color palette based on number of categories
        if type_colors is None:
            type_colors = type_20_colors if len(__hue_order__) <= 20 else type_50_colors

        colors = {}

        # Create a copy of hue column for legend renaming if needed
        if legend is not None:
            df.loc[:, "__hue__"] = df[hue].copy()

        # Assign colors to each category
        i = 0

        for elem in __hue_order__:
            if legend is not None:
                # Rename categories according to legend mapping
                df.loc[df[df["__hue__"] == elem].index, "__hue__"] = legend[elem]
                colors.update(
                    {legend[elem]: type_colors[start_color_index + i * color_step_size + __hue_order__.index(elem)]}
                )
            else:
                colors.update(
                    {
                        elem: type_colors[start_color_index + i * color_step_size + __hue_order__.index(elem)]
                    }
                )
            i += 1

        # Determine hue order for plotting
        if legend is not None:
            if hue_order is None:
                hue_order = list(np.sort(list(set(df["__hue__"]))))
        else:
            if hue_order is None:
                hue_order = __hue_order__

        # Create scatter plot with categorical colors
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            ax=ax,
            edgecolor=edge_color,
            palette=colors,
            hue="__hue__" if legend is not None else hue,
            hue_order=hue_order,
            s=size,
            **kwargs
        )

        # Add text annotations at centroid positions if requested
        if is_text:

            df_anno = df[[hue, x, y]].groupby(hue, as_index=False).mean()

            for txt, i, j in zip(df_anno[hue], df_anno[x], df_anno[y]):
                plt.annotate(
                    txt,
                    xy=(i, j),
                    xytext=(-10, 0),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="k",
                        lw=text_lw,
                        alpha=0.8
                    ),
                    fontsize=text_fontsize
                )

    # Remove scales and labels on the coordinate axis
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plot_end(title, x_name, y_name, output, show, close)

    return ax


def scatter_3d(
    df: DataFrame,
    x: str,
    y: str,
    z: str,
    hue: str = None,
    x_name: str = None,
    y_name: str = None,
    z_name: str = None,
    title: str = None,
    width: float = 7,
    height: float = 7,
    elev: float = 30,
    azim: float = -60,
    is_add_legend: bool = True,
    cmap: Union[str, ListedColormap] = 'tab20',
    font_size: int = 14,
    text_font_size: int = 9,
    edge_color: str = None,
    size: Union[float, collection] = 0.1,
    legend_name: str = None,
    is_add_max_label: bool = False,
    text_left_offset: float = 0.5,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Create a 3D scatter plot with customizable aesthetics.
    
    Parameters
    ----------
    df : DataFrame
        Input data containing x, y, z coordinates
    x : str
        Column name for x-axis values
    y : str
        Column name for y-axis values
    z : str
        Column name for z-axis values
    hue : str, optional
        Column name for color grouping
    x_name : str, optional
        Label for x-axis
    y_name : str, optional
        Label for y-axis
    z_name : str, optional
        Label for z-axis
    title : str, optional
        Plot title
    width : float, default 7
        Figure width in inches
    height : float, default 7
        Figure height in inches
    elev : float, default 30
        Elevation angle for 3D view
    azim : float, default -60
        Azimuth angle for 3D view
    is_add_legend : bool, default True
        Whether to add legend
    cmap : Union[str, ListedColormap], default 'tab20'
        Colormap for coloring
    text_font_size : int, default 9
    font_size : int, default 14
        Font size for labels and title
    edge_color : str, optional
        Edge color for scatter points
    size : Union[float, collection], default 0.1
        Size of scatter points
    legend_name : str, optional
        Title for legend
    is_add_max_label : bool, default False
        Whether to add label for maximum z value point
    text_left_offset : float, default 0.5
        Horizontal offset for max value label
    output : path, optional
        Output file path
    show : bool, default True
        Whether to display the plot
    close : bool, default False
        Whether to close the figure after saving
    **kwargs : Any
        Additional arguments passed to ax.scatter
    """

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(projection='3d')

    hue_cat = None

    if hue is not None:
        hue_cat = pd.Categorical(df[hue])

    scatter_ = ax.scatter(
        df[x],
        df[y],
        df[z],
        c=hue_cat.codes if hue is not None else None,
        cmap=cmap,
        s=size,
        edgecolors=edge_color,
        **kwargs
    )

    # angle of view
    ax.view_init(elev=elev, azim=azim)

    if x_name is not None:
        ax.set_xlabel(x_name, fontsize=font_size)

    if y_name is not None:
        ax.set_ylabel(y_name, fontsize=font_size)

    if z_name is not None:
        ax.set_zlabel(z_name, fontsize=font_size)

    if title is not None:
        ax.set_title(title, fontsize=font_size)

    if is_add_legend and hue is not None:
        unique_types = hue_cat.categories
        legend_elements = [
            plt.Line2D(
                [0], [0], marker='o', color='w', label=type_,
                markerfacecolor=scatter_.cmap(scatter_.norm(i))
            )
            for i, type_ in enumerate(unique_types)
        ]

        ax.legend(handles=legend_elements, title=legend_name, bbox_to_anchor=(1.1, 1))

    if is_add_max_label:
        max_idx = df[z].idxmax()
        max_x = df.loc[max_idx, x]
        max_y = df.loc[max_idx, y]
        max_value = df.loc[max_idx, z]

        # Add text label at the position of the maximum value point
        ax.text(
            max_x - text_left_offset,
            max_y,
            max_value,
            f'({max_x}, {max_y}): {max_value:.3f}',
            fontsize=text_font_size,
            color='red',
            ha='left'
        )

    plot_end(None, None, None, output, show, close)

    return ax


def scatter_element(
    adata: AnnData,
    title: str = None,
    element_name: str = "All",
    layers: Union[None, collection] = None,
    coordinates: Tuple[str, str] = ("UMAP1", "UMAP2"),
    cmap: str = "viridis",
    x_name: str = None,
    y_name: str = None,
    number: bool = True,
    edge_color: str = None,
    bar_title: str = None,
    size: Union[float, collection] = 1.0,
    text_fontsize: float = 7,
    start_color_index: int = 0,
    color_step_size: int = 0,
    type_colors: collection = None,
    is_text: bool = False,
    legend: dict = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> None:
    """
    Plot trait data scatter plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing trait/disease scores and cell metadata
    title : str, optional
        Title prefix for the plot
    bar_title : str, optional
        Label for colorbar when number=True
    element_name : str, default "All"
        Name of trait/disease to plot, or "All" to plot all traits
    layers : Union[None, collection], optional
        List of layer names to plot from trait_adata.layers
    coordinates : Tuple[str, str], default ("UMAP1", "UMAP2")
        Column names for x and y coordinates in trait_adata.obs
    cmap : str, default "viridis"
        Colormap for continuous coloring
    x_name : str, optional
        Label for x-axis
    y_name : str, optional
        Label for y-axis
    number : bool, default True
        Whether to use continuous color scale for trait scores
    edge_color : str, optional
        Edge color for scatter points
    size : Union[float, collection], default 1.0
        Size of scatter points
    text_fontsize : float, default 7
        Font size for annotation text
    start_color_index : int, default 0
        Starting index in color palette
    color_step_size : int, default 0
        Step size for color selection
    type_colors : collection, optional
        Custom color palette
    is_text : bool, default False
        Whether to add text annotations
    legend : dict, optional
        Mapping to rename hue categories
    output : path, optional
        Output directory path for saving plots
    show : bool, default True
        Whether to display the plot
    close : bool, default False
        Whether to close the figure after saving
    **kwargs : Any
        Additional arguments passed to scatter_base
    """
    data: AnnData = adata.copy()

    # judge layers
    adata_layers = list(data.layers)

    if layers is not None and len(layers) != 0:

        for layer in layers:

            if layer not in adata_layers:
                log.error(f"The `layers` parameter ({layer}) needs to include in `adata.layers`")
                raise ValueError(f"The `layers` parameter ({layer}) needs to include in `adata.layers`")

    def element_plot(element_: str, df_: DataFrame, layer_: str = None, new_data_: AnnData = None) -> None:
        """
        show plot
        :param element_: trait name
        :param df_:
        :param layer_: layer
        :param new_data_:
        :return: None
        """
        log.info(f"Plotting scatter {element_}")
        # get gene score
        element_score = new_data_[:, element_].to_df()
        element_score = element_score.rename_axis("__barcode__")
        element_score.reset_index(inplace=True)
        df_ = df_.rename_axis("__barcode__")
        df_.reset_index(inplace=True)
        # trait_score.rename_axis("index")
        df = df_.merge(element_score, on="__barcode__", how="left")
        # Sort gene scores from small to large
        df.sort_values([element_], inplace=True)
        scatter(
            df,
            x=coordinates[0],
            y=coordinates[1],
            hue=element_,
            title=f"{title} {element_}" if title is not None else title,
            legend=legend,
            cmap=cmap,
            number=number,
            size=size,
            x_name=x_name,
            y_name=y_name,
            bar_title=bar_title,
            type_colors=type_colors,
            text_fontsize=text_fontsize,
            start_color_index=start_color_index,
            color_step_size=color_step_size,
            edge_color=edge_color,
            is_text=is_text,
            output=os.path.join(
                output, f"{element_}_{layer_}_score.pdf" if layer_ is not None else f"{element_}_score.pdf"
            ) if output is not None else None,
            show=show,
            close=close,
            **kwargs
        )

    def handle_plot(layer_: str = None):
        # DataFrame
        df: DataFrame = data.obs.copy()
        df.rename_axis("index", inplace=True)
        element_list: list = list(data.var_names)

        # judge element
        if element_name != "All" and element_name not in element_list:
            log.error(f"The {element_name} element is not in the element list (adata.var_names)")
            raise ValueError(f"The {element_name} element is not in the element list (adata.var_names)")

        new_data: AnnData = AnnData(data.layers[layer], var=data.var, obs=data.obs) if layer_ is not None else data

        # plot
        if element_name == "All":
            for element in element_list:
                element_plot(element, df, layer_, new_data)
        else:
            element_plot(element_name, df, layer_, new_data)

    if layers is None or len(layers) == 0:
        handle_plot()
    else:
        for layer in layers:
            log.info(f"Start {layer}")
            handle_plot(layer)


def volcano(
    df: DataFrame,
    x: str = "Log2(Fold change)",
    y: str = "-Log10(P value)",
    hue: str = "type",
    size: int = 3,
    palette: Optional[list] = None,
    y_min: float = 0,
    axh_value: float = -np.log10(1e-3),
    axv_left_value: float = -1,
    axv_right_value: float = 1,
    title: str = None,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Plot volcano plot.

    Parameters
    ----------
    df : DataFrame
        Data frame.
    x : str, optional
        X-axis.
    y : str, optional
        Y-axis.
    hue : str, optional
        Hue.
    size : int, optional
        Size.
    palette : Optional[list], optional
        Palette.
    y_min : float, optional
        Y-min.
    axh_value : float, optional
        Axh-value.
    axv_left_value : float, optional
        Axv-left-value.
    axv_right_value : float, optional
        Axv-right-value.
    title : str, optional
        Title.
    x_name : Optional[str], optional
        X-name.
    y_name : Optional[str], optional
        Y-name.
    output : path, optional
        Output.
    show : bool, optional
        Show to display the plot.
    close : bool, optional
        Close to close the figure after saving.
    kwargs : Any, optional
        Additional keyword arguments passed to sns.scatterplot.
    """

    fig, ax = plot_start()

    if palette is None:
        palette = ["#01c5c4", "#686d76", "#ff414d"]

    sns.set_theme(style="ticks")
    sns.set_palette(sns.color_palette(palette))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=size, ax=ax, **kwargs)
    ax.set_ylim(y_min, max(df[y]) * 1.1)

    plt.axhline(axh_value, color='grey', linestyle='--')
    plt.axvline(axv_left_value, color='grey', linestyle='--')
    plt.axvline(axv_right_value, color='grey', linestyle='--')

    plot_end(title, x_name, y_name, output, show, close)

    return ax


def manhattan(
    df: DataFrame,
    y: str = "pp",
    groupby: str = "chr",
    label: str = "rsId",
    size: int = 30,
    labels: Optional[list] = None,
    colors: Optional[list] = None,
    title: str = None,
    is_sort: bool = True,
    line_width: float = 0.5,
    y_round: int = 3,
    x_name: Optional[str] = "Chromosome",
    y_name: Optional[str] = "pp",
    y_limit: Tuple[float, float] = (0, 1),
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Create a Manhattan plot for causal variant visualization across chromosomes.
    
    Parameters
    ----------
    df : DataFrame
        Input data containing variant information with chromosome and position data
    y : str, default "pp"
        Column name for y-axis values (typically posterior probability or p-value)
    groupby : str, default "chr"
        Column name for chromosome identifiers
    label : str, default "rsId"
        Column name for variant labels/identifiers
    size : int, default 30
        Size of scatter points
    labels : Optional[list], optional
        List of specific variant labels to annotate on the plot
    colors : Optional[list], optional
        Custom color palette for different chromosomes
    title : str, optional
        Plot title
    is_sort : bool, default True
        Whether to sort data by chromosome
    line_width : float, default 0.5
        Width of separator lines between chromosomes and grid lines
    y_round : int, default 3
        Number of decimal places for y-value annotations
    x_name : Optional[str], default "Chromosome"
        Label for x-axis
    y_name : Optional[str], default "pp"
        Label for y-axis
    y_limit : Tuple[float, float], default (0, 1)
        Y-axis limits for the plot
    output : path, optional
        Output file path
    show : bool, default True
        Whether to display the plot
    close : bool, default False
        Whether to close the figure after saving
    **kwargs : Any
        Additional arguments passed to ax.axvline
    """
    df[groupby] = df[groupby].astype('category')

    if is_sort:
        df = df.sort_values(groupby)

    df['ind'] = range(len(df))
    df_grouped = df.groupby(groupby)

    if colors is None:
        colors = type_set_colors.copy()
        colors.extend(list(type_20_colors))

    fig, ax = plot_start()

    x_labels = []
    x_labels_pos = []
    # Track the last index to draw lines between chromosomes
    last_ind = 0

    groupby_unique = df[groupby].unique()

    for num, (name, group) in enumerate(df_grouped):

        if name not in groupby_unique:
            continue

        group.plot(kind='scatter', x='ind', y=y, color=colors[num], s=size, ax=ax)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))

        # Draw a vertical line between chromosomes
        if num > 0:
            # Skip the first chromosome
            ax.axvline(x=last_ind + 0.5, color='gray', linestyle='--', linewidth=line_width, **kwargs)

        # Label specific mutations
        if labels is not None:
            for index, row in group.iterrows():
                if row[label] in labels:
                    ax.text(row['ind'], row[y], row[label], ha='left', va='bottom')
                    ax.text(row['ind'], row[y], f"{y}={round(row[y], y_round)}", ha='left', va='top')

        last_ind = group['ind'].iloc[-1]

    # add grid
    ax.grid(axis="y", linestyle="--", linewidth=line_width, color="gray")
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)

    # Hide the borders above and to the right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim([0, len(df)])
    ax.set_ylim(y_limit)

    plot_end(title, x_name, y_name, output, show, close)

    return ax


def pseudo_time_score(
    df: DataFrame,
    x: str,
    y: str,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    alpha: float = 0.65,
    line_width: float = 1.0,
    step_length: int = 5,
    polyorder: int = 1,
    colors = None,
    line_color: str = "black",
    size: Union[float, collection] = 1.0,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Create a scatter plot showing pseudo-time scores with a smoothed trend line.
    
    Parameters
    ----------
    df : DataFrame
        Input data containing pseudo-time and score values
    x : str
        Column name for pseudo-time values (x-axis)
    y : str
        Column name for score values (y-axis)
    x_name : str, optional
        Label for x-axis
    y_name : str, optional
        Label for y-axis
    title : str, optional
        Plot title
    alpha : float, default 0.65
        Transparency of scatter points
    line_width : float, default 1.5
        Width of the smoothed trend line
    step_length : int, default 5
        Step length for determining Savitzky-Golay filter window size
    colors : default None
    polyorder : int, default 1
        Polynomial order for Savitzky-Golay filter
    line_color : str, default black
    size : Union[float, collection], default 1.0
        Size of scatter points
    output : path, optional
        Output file path
    show : bool, default True
        Whether to display the plot
    close : bool, default False
        Whether to close the figure after saving
    **kwargs : Any
        Additional arguments passed to ax.scatter
    """
    from scipy.signal import savgol_filter

    df = df.sort_values(by=x)

    fig, ax = plot_start()

    pseudo_times = df[x].values
    scores = df[y].values

    x_len = len(pseudo_times)

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, x_len))

    ax.scatter(
        pseudo_times,
        scores,
        c=colors,
        alpha=alpha,
        s=size,
        **kwargs
    )

    window_length = max(3, int(x_len / step_length))

    if window_length % 2 == 0:
        window_length += 1

    smoothed_scores = savgol_filter(scores, window_length=window_length, polyorder=polyorder)

    ax.plot(pseudo_times, smoothed_scores, color=line_color, linewidth=line_width)

    plt.tight_layout()

    plot_end(title, x_name, y_name, output, show, close)

    return ax

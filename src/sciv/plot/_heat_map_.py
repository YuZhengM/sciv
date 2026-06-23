# -*- coding: UTF-8 -*-

from typing import Optional, Union, Tuple, Any

import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PyComplexHeatmap import HeatmapAnnotation, anno_simple, ClusterMapPlotter, anno_label, anno_barplot
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pandas import DataFrame
import seaborn as sns

from .. import util as ul
from ..util import path, type_20_colors, type_50_colors, plot_end, plot_start

__name__: str = "plot_heat_map"

log = ul.log(__name__, "ERROR")


def heatmap_annotation(
    adata: AnnData,
    layer: Optional[str] = None,
    title: Optional[str] = None,
    label: str = "value",
    row_name: Optional[str] = None,
    col_name: Optional[str] = None,
    row_names: Optional[str] = None,
    col_names: Optional[str] = None,
    row_anno_label: bool = False,
    col_anno_label: bool = False,
    row_anno_text: bool = False,
    col_anno_text: bool = False,
    row_legend: bool = False,
    col_legend: bool = False,
    row_show_names: bool = False,
    col_show_names: bool = False,
    row_cluster: bool = False,
    col_cluster: bool = False,
    cluster_method: str = "average",
    cluster_metric: str = "correlation",
    row_names_side: str = "left",
    col_names_side: str = "bottom",
    label_size: float = 9,
    fontsize: float = 9,
    level_bar_height: float = None,
    anno_specific_labels: list = None,
    x_label_rotation: float = 245,
    y_label_rotation: float = 0,
    row_color_start_index: int = 0,
    col_color_start_index: int = 10,
    row_split: Union[int, pd.Series] = None,
    col_split: Union[int, pd.Series] = None,
    row_split_order: Union[list, str] = None,
    col_split_order: Union[list, str] = None,
    row_split_gap: float = 0.5,
    col_split_gap: float = 0.2,
    frac: float = 0.2,
    relpos: Tuple = (0, 1),
    anno_label_height: Optional[float] = None,
    selected_anno_label_height: float = 2.5,
    category_height: Optional[float] = 2.5,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    row_score_name: Optional[str] = None,
    cmap: str = "Oranges",
    is_sort: bool = True,
    show: bool = False,
    close: bool = True,
    output: path = None,
    **kwargs
) -> tuple[Figure, Any]:
    """
    Generate a heatmap with row and column annotations.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing the data matrix and metadata.
    layer : Optional[str], default None
        Layer name in adata.layers to use for plotting. If None, uses adata.X.
    title : Optional[str], default None
        Title of the figure.
    label : str, default "value"
        Label for the heatmap color bar.
    row_name : Optional[str], default None
        Column name in adata.obs for row annotations.
    col_name : Optional[str], default None
        Column name in adata.var for column annotations.
    row_names : Optional[str], default None
        Column name in adata.obs to use as row index labels.
    col_names : Optional[str], default None
        Column name in adata.var to use as column index labels.
    row_anno_label : bool, default False
        Whether to display merged labels for row annotations.
    col_anno_label : bool, default False
        Whether to display merged labels for column annotations.
    row_anno_text : bool, default False
        Whether to display text labels on row annotation bars.
    col_anno_text : bool, default False
        Whether to display text labels on column annotation bars.
    row_legend : bool, default False
        Whether to show legend for row annotations.
    col_legend : bool, default False
        Whether to show legend for column annotations.
    row_show_names : bool, default False
        Whether to display row names (index labels) on the heatmap.
    col_show_names : bool, default False
        Whether to display column names (index labels) on the heatmap.
    row_cluster : bool, default False
        Whether to perform hierarchical clustering on rows.
    col_cluster : bool, default False
        Whether to perform hierarchical clustering on columns.
    cluster_method : str, default "average"
        Linkage method for hierarchical clustering (e.g., "average", "single", "complete").
    cluster_metric : str, default "correlation"
        Distance metric for hierarchical clustering (e.g., "correlation", "euclidean").
    row_names_side : str, default "left"
        Side to display row names ("left" or "right").
    col_names_side : str, default "bottom"
        Side to display column names ("top" or "bottom").
    label_size : float, default 9
        Font size for row and column name labels.
    fontsize : float, default 9
        Font size for axis titles.
    level_bar_height : float, default None
        Height of the association score bar plot annotation.
    anno_specific_labels : list, default None
        List of specific row labels to highlight in the annotation.
    x_label_rotation : float, default 245
        Rotation angle for x-axis labels (column names).
    y_label_rotation : float, default 0
        Rotation angle for y-axis labels (row names).
    row_color_start_index : int, default 0
        Starting index in the color palette for row annotations.
    col_color_start_index : int, default 10
        Starting index in the color palette for column annotations.
    row_split : Union[int, pd.Series], default None
        Number of clusters or grouping series for splitting rows.
    col_split : Union[int, pd.Series], default None
        Number of clusters or grouping series for splitting columns.
    row_split_order : Union[list, str], default None
        Order for row splits or 'cluster_between_groups' for auto-clustering.
    col_split_order : Union[list, str], default None
        Order for column splits or 'cluster_between_groups' for auto-clustering.
    row_split_gap : float, default 0.5
        Gap size between row splits in mm.
    col_split_gap : float, default 0.2
        Gap size between column splits in mm.
    frac : float, default 0.2
        Fraction parameter for annotation label positioning.
    relpos : Tuple, default (0, 1)
        Relative position for annotation labels.
    anno_label_height : Optional[float], default None
        Height of the annotation label bar.
    selected_anno_label_height : float, default 2.5
        Height of the selected annotation label bar.
    category_height : Optional[float], default 2.5
        Height of the category annotation bar.
    x_name : Optional[str], default None
        Label for the x-axis.
    y_name : Optional[str], default None
        Label for the y-axis.
    row_score_name : str, default "association_score"
        Column name in adata.obs for the association score bar plot.
    cmap : str, default "Oranges"
        Colormap for the heatmap.
    is_sort : bool, default True
        Whether to sort rows and columns before plotting.
    show : bool, default True
        Whether to display the figure.
    close : bool, default False
        Whether to close the figure after saving.
    output : path, default None
        File path to save the figure. If None, figure is not saved.
    **kwargs
        Additional keyword arguments passed to ClusterMapPlotter.

    Returns
    -------
    None
        Displays or saves the heatmap figure.
    """
    log.info("Start plotting the heatmap")
    fig, ax = plot_start()

    data = adata.copy()

    # judge layers
    if layer is not None:

        if layer not in list(data.layers):
            log.error("The value of the `layer` parameter must be one of the keys in `adata.layers`.")
            raise ValueError(f"The `{layer}` parameter needs to include in `adata.layers`")

        data.X = data.layers[layer]

    if is_sort:
        data = data[data.obs.sort_values(row_name).index, data.var.sort_values(col_name).index]

    # DataFrame
    df: DataFrame = data.to_df()

    row_anno: DataFrame = data.obs.copy()

    col_anno: DataFrame = data.var.copy()

    if row_names is not None:
        df.index = data.obs[row_names].astype(str)
        row_anno.index = data.obs[row_names].astype(str)

    if col_names is not None:
        df.columns = data.var[col_names].astype(str)
        col_anno.index = data.var[col_names].astype(str)

    if row_name is not None:
        row_colors = type_20_colors[row_color_start_index:] if len(
            list(set(row_anno[row_name]))) + row_color_start_index <= 20 else type_50_colors[row_color_start_index:]
    else:
        row_colors = "cmap50"

    if col_name is not None:
        col_colors = type_20_colors[col_color_start_index:] if len(
            list(set(col_anno[col_name]))) + col_color_start_index <= 20 else type_50_colors[col_color_start_index:]
    else:
        col_colors = "cmap50"

    df_rows = None
    if anno_specific_labels is not None:
        df_rows = df.apply(lambda x: x.name if x.name in anno_specific_labels else None, axis=1)
        df_rows.name = "Selected"

    # noinspection PyTypeChecker
    row_ha = HeatmapAnnotation(
        label=anno_label(
            row_anno[row_name], cmap=ListedColormap(row_colors), merge=True, height=anno_label_height
        ) if row_anno_label and row_name else None,
        RowCategory=anno_simple(
            row_anno[row_name],
            cmap=ListedColormap(row_colors),
            height=category_height,
            legend=row_legend,
            add_text=row_anno_text,
            text_kws=dict(color="black", rotation=0, fontsize=label_size),
        ) if row_name is not None else None,
        axis=0,
        verbose=0,
        legend_gap=5,
        hgap=0.5,
        label_kws=dict(color="black", rotation=90, horizontalalignment="left")
    )

    # noinspection PyTypeChecker
    row_ha_right = HeatmapAnnotation(
        AssociationScore=anno_barplot(row_anno[[row_score_name]], legend=True, height=level_bar_height,
                                      **dict(edgecolor='none')) if row_score_name in row_anno.columns else None,
        selected=anno_label(df_rows, relpos=relpos, frac=frac, fontsize=label_size,
                            height=selected_anno_label_height) if anno_specific_labels is not None else None,
        axis=0,
        verbose=0,
        legend_gap=5,
        hgap=0.5,
        label_kws=dict(color="black", rotation=90, horizontalalignment="left")
    )

    col_ha_args = {"rotation": 90}
    # noinspection PyTypeChecker
    col_ha = HeatmapAnnotation(
        label=anno_label(
            col_anno[col_name], cmap=ListedColormap(col_colors), merge=True, height=anno_label_height, **col_ha_args
        ) if col_anno_label and col_name else None,
        ColCategory=anno_simple(
            col_anno[col_name],
            cmap=ListedColormap(col_colors),
            height=category_height,
            add_text=col_anno_text,
            legend=col_legend,
            text_kws={'fontsize': label_size}
        ) if col_name is not None else None,
        axis=1,
        verbose=0,
        legend_gap=5,
        hgap=0.5,
        label_side='left',
        label_kws=dict(color="black", rotation=0, horizontalalignment="right")
    )

    """
    It is worth noting here that, `row_cluster_metric="correlation"`, When the default parameter 
    `row_cluster_metric` in method `ClusterMapPlotter` is passed into method `distance.pdist`, 
    that is `metric='correlation'`, and this method derives from this 
    `https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html`,
    it can be inferred that there is a division formula in one step, which may result in the possibility of `NA`.
    
    For example, in this scATAC-seq data, if there are two or more traits without any intersection, 
    the denominator will appear as zero.
    
    Therefore, use the `"median"` value for parameter `row_cluster_method`
    Therefore, use the `"euclidean"` value for parameter `row_cluster_metric`
    """

    ClusterMapPlotter(
        data=df,
        top_annotation=col_ha if col_name is not None else None,
        left_annotation=row_ha if row_name is not None else None,
        right_annotation=row_ha_right if anno_specific_labels is not None or row_score_name in row_anno.columns else None,
        label=label,
        row_cluster_method=cluster_method,
        row_cluster_metric=cluster_metric,
        col_cluster_method=cluster_method,
        col_cluster_metric=cluster_metric,
        show_rownames=row_show_names,
        show_colnames=col_show_names,
        row_names_side=row_names_side,
        col_names_side=col_names_side,
        col_split=col_split,
        row_split=row_split,
        row_split_order=row_split_order,
        col_split_order=col_split_order,
        col_split_gap=col_split_gap,
        row_split_gap=row_split_gap,
        xticklabels_kws=dict(labelrotation=x_label_rotation, labelcolor='black', labelsize=label_size),
        yticklabels_kws=dict(labelrotation=y_label_rotation, labelcolor='black', labelsize=label_size),
        cmap=cmap,
        tree_kws={'row_cmap': 'Dark2'},
        xlabel=x_name,
        ylabel=y_name,
        xlabel_kws=dict(color='black', fontsize=fontsize),
        ylabel_kws=dict(color='black', fontsize=fontsize),
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        col_dendrogram=col_cluster,
        row_dendrogram=row_cluster,
        **kwargs
    )

    plot_end(fig, title, None, None, output, show, close)

    return fig, ax


def heatmap(
    adata: AnnData,
    layer: str = None,
    title: Optional[str] = None,
    annot: bool = False,
    square: bool = True,
    is_cluster: bool = False,
    cmap: str = "Oranges",
    line_widths: float = 1,
    fmt: str = ".2f",
    rotation: float = 65,
    x_name: str = None,
    y_name: str = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Figure, Any]:
    """
    Generate a simple heatmap using seaborn.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing the data matrix.
    layer : str, default None
        Layer name in adata.layers to use for plotting. If None, uses adata.X.
    title : Optional[str], default None
        Title of the figure.
    annot : bool, default False
        Whether to annotate each cell with its numeric value.
    square : bool, default True
        Whether to make cells square-shaped.
    is_cluster : bool, default False
        Whether to perform hierarchical clustering (uses clustermap instead of heatmap).
    cmap : str, default "Oranges"
        Colormap for the heatmap.
    line_widths : float, default 1
        Width of the lines that divide cells.
    fmt : str, default ".2f"
        String formatting code for annotations.
    rotation : float, default 65
        Rotation angle for x-axis labels.
    x_name : str, default None
        Label for the x-axis.
    y_name : str, default None
        Label for the y-axis.
    output : path, default None
        File path to save the figure. If None, figure is not saved.
    show : bool, default True
        Whether to display the figure.
    close : bool, default False
        Whether to close the figure after saving.
    **kwargs : Any
        Additional keyword arguments passed to seaborn heatmap or clustermap.

    Returns
    -------
    None
        Displays or saves the heatmap figure.
    """
    fig, ax = plot_start()

    data = adata.copy()

    # judge layers
    if layer is not None:

        if layer not in list(data.layers):
            log.error("The value of the `layer` parameter must be one of the keys in `adata.layers`.")
            raise ValueError("The value of the `layer` parameter must be one of the keys in `adata.layers`.")

        data.X = data.layers[layer]

    # DataFrame
    log.info(f"to DataFrame")
    df: DataFrame = data.to_df()
    # seaborn
    heat_map: Axes = sns.clustermap(data=df, square=square, annot=annot, cmap=cmap, fmt=fmt, **kwargs) \
        if is_cluster else \
        sns.heatmap(data=df, square=square, annot=annot, cmap=cmap, linewidths=line_widths, fmt=fmt, **kwargs)

    if not is_cluster:
        plt.setp(heat_map.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor")
    else:
        # noinspection PyUnresolvedReferences
        plt.setp(heat_map.ax_heatmap.get_xticklabels(), rotation=rotation)

    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, ax

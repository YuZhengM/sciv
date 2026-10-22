# -*- coding: UTF-8 -*-

from typing import Tuple, Union, Any, Literal

import pandas as pd
from pandas import DataFrame
import seaborn as sns

from .. import util as ul
from ..util import path, plot_end, plot_start, plot_color_types

__name__: str = "plot_box"

log = ul.log(__name__, "ERROR")


def box(
    df: DataFrame,
    x: str = "clusters",
    y: str = "value",
    x_name: str = None,
    y_name: str = "value",
    palette: Union[Tuple, list, dict] = None,
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
):
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

    plot_end(title, x_name, y_name, output, show, close)

    return ax


def box_group(
    df: DataFrame,
    x: str = "metrics",
    y: str = "value",
    hue: str = "clusters",
    x_name: str = None,
    y_name: str = "value",
    legend: str = None,
    legend_list: list = None,
    cmap: str = "set",
    line_width: float = 0.6,
    rotation: float = 0,
    width: float = 0.85,
    colors: Union[list, dict] = None,
    start_color_index: int = 0,
    color_step_size: int = 1,
    dodge: bool = True,
    title: str = None,
    whis: float = 1.5,
    show_fliers: bool = False,
    show_errorbar: bool = True,
    errorbar_color: str = "black",
    errorbar_capsize: float = 0.04,
    y_limit: Tuple[float, float] = None,
    order_names: list = None,
    hue_order: list = None,
    output: str = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Create a grouped box plot with dashed grids, colored boxes
    and mean +/- error bars.
    """
    # ---------- 参数校验 ----------
    df_columns = list(df.columns)

    for col in (x, y, hue):
        if col not in df_columns:
            log.error(f"The `{col}` parameter must be in the `df` parameter data column name ({df_columns})")
            raise ValueError(f"The `{col}` parameter must be in the `df` parameter data column name ({df_columns})")

    # ---------- 数据筛选 ----------
    # FIX: 筛选和绘图统一用 new_data，原代码 boxplot 用的是 df（未筛选）
    if legend_list is not None:
        new_data: DataFrame = df[df[hue].isin(legend_list)].copy()
    else:
        new_data: DataFrame = df.copy()

    if legend is None:
        legend = "category"

    new_data[legend] = new_data[hue].astype(str)

    # ---------- hue 顺序 ----------
    if hue_order is not None:
        new_data[legend] = pd.Categorical(new_data[legend], categories=hue_order, ordered=True)
        new_data = new_data.sort_values(by=legend)

    hue_types = new_data[legend].unique().tolist()

    if order_names is None:
        order_names = new_data[x].unique().tolist()

    # ---------- 颜色 ----------
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
        if "color" in new_data.columns:
            # FIX: 按 hue 类型取颜色，保证同一 hue 颜色一致（原代码按行 zip 可能冲突）
            color_map = dict(zip(new_data[legend], new_data["color"]))
            palette = [color_map[h] for h in hue_types]
        else:
            palette = []
            for i, hue_type in enumerate(hue_types):
                # FIX: 原公式 start_color_index + _index_*step + _index_ 重复加了 _index_
                _idx = start_color_index + i * color_step_size
                palette.append(plot_color_types[cmap][_idx])

    palette_dict = dict(zip(hue_types, palette))

    # ---------- 绘图 ----------
    fig, ax = plot_start()

    if y_limit is not None:
        ax.set_ylim(y_limit)

    flierprops = {'marker': 'o', 'markersize': 0.2}
    flierprops.update(kwargs.pop("flierprops", {}))

    boxprops = {'edgecolor': 'black', 'linewidth': line_width}
    boxprops.update(kwargs.pop("boxprops", {}))

    whiskerprops = {'color': 'black', 'linewidth': line_width}
    whiskerprops.update(kwargs.pop("whiskerprops", {}))

    capprops = {'color': 'black', 'linewidth': line_width}
    capprops.update(kwargs.pop("capprops", {}))

    medianprops = {'color': 'black', 'linewidth': line_width}
    medianprops.update(kwargs.pop("medianprops", {}))

    sns.boxplot(
        data=new_data,
        x=x,
        y=y,
        hue=legend,
        order=order_names,
        hue_order=hue_types,
        palette=palette_dict,
        width=width,
        dodge=dodge,
        whis=whis,
        showfliers=show_fliers,
        flierprops=flierprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        ax=ax,
        **kwargs
    )

    # ---------- 误差条（均值 ± 标准误）----------
    if show_errorbar:
        n_hue = len(hue_types)

        offsets = [
            -width / 2 + width * (i + 0.5) / n_hue
            for i in range(n_hue)
        ] if dodge else [0]

        for i, h in enumerate(hue_types):
            for j, x_cat in enumerate(order_names):
                # FIX: 用 new_data 和 legend 列，与筛选后的数据保持一致
                sub = new_data[(new_data[x] == x_cat) & (new_data[legend] == h)][y]
                if len(sub) == 0:
                    continue

                mean = sub.mean()
                sem = sub.std() / (len(sub) ** 0.5) if len(sub) > 1 else 0

                ax.errorbar(
                    j + offsets[i],
                    mean,
                    yerr=sem,
                    fmt='none',
                    ecolor=errorbar_color,
                    elinewidth=line_width,
                    capsize=errorbar_capsize * 10,
                    capthick=line_width,
                    zorder=10
                )

        # FIX: 原代码 title=hue_types（传了列表），且 legend 变量被覆盖
        #      seaborn 已自动生成图例，这里只需设置标题
        legend_handle = ax.get_legend()
        if legend_handle is not None and legend is not None:
            legend_handle.set_title(legend)
            legend_handle.set_frame_on(False)

    # ---------- 网格 / 坐标轴 ----------
    ax.yaxis.grid(True, linestyle='--', linewidth=line_width, color='gray', alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_xticks(range(len(order_names)))
    ax.set_xticklabels(labels=order_names, rotation=rotation)

    ax.set_xlabel(x_name if x_name is not None else x)
    ax.set_ylabel(y_name)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)

    plot_end(title, None, y_name, output, show, close)

    return ax

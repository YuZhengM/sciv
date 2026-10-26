# -*- coding: UTF-8 -*-

from typing import Tuple, Union, Literal, Any, List, Optional

import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from pandas import DataFrame
import seaborn as sns

from .. import util as ul
from ..tool import get_stat_result
from ..util import path, plot_start, plot_end, test_method_type

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
    palette: Union[Tuple, list, dict] = None,
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
):
    """
    Plot violin plot (or other categorical plots).

    Parameters
    ----------
    df : DataFrame
        Input data.
    value : str, optional
        Value column (Y-axis).
    x_name : str, optional
        Label for X-axis.
    y_name : str, optional
        Label for Y-axis.
    kind : _Kind, optional
        Kind of plot (e.g., "violin", "box", "bar").
    groupby : str, optional
        Column name for grouping (X-axis).
    palette : Union[Tuple, list], optional
        Color palette.
    hue : str, optional
        Column name for hue grouping.
    rotation : float, optional
        Rotation angle for x-axis labels.
    line_width : float, optional
        Line width for axes and plot elements.
    title : str, optional
        Plot title.
    split : bool, optional
        Whether to split violins (only works when kind='violin').
    is_sort : bool, optional
        Whether to sort groups by median value.
    order_names : list, optional
        Specific order for groups.
    output : path, optional
        Output file path.
    show : bool, optional
        Whether to display the plot.
    close : bool, optional
        Whether to close the plot figure.
    kwargs : Any, optional
        Additional keyword arguments for sns.catplot.
    """
    # judge columns
    df_columns = list(df.columns)

    if value not in df_columns:
        log.error(f"The `value` ({value}) parameter must be in the `df` parameter data column name ({df_columns})")
        raise ValueError(
            f"The `value` ({value}) parameter must be in the `df` parameter data column name ({df_columns})"
        )

    if hue is not None and hue not in df_columns:
        log.error(f"The `hue` ({hue}) parameter must be in the `df` parameter data column name ({df_columns})")
        raise ValueError(f"The `hue` ({hue}) parameter must be in the `df` parameter data column name ({df_columns})")

    if groupby not in df_columns:
        log.error(f"The `groupby` ({groupby}) parameter must be in the `df` parameter data column name ({df_columns})")
        raise ValueError(
            f"The `groupby` ({groupby}) parameter must be in the `df` parameter data column name ({df_columns})"
        )

    # Prepare sorting and colors
    group_columns = [groupby]

    if order_names is not None:
        # 优先使用用户指定的顺序
        y_names = order_names
    elif is_sort:
        # 按中位数排序
        # 注意：这里只聚合数值列计算中位数，不涉及 color
        median_df = df.groupby(group_columns, observed=True, as_index=False)[value].median()
        median_df.sort_values([value], ascending=False, inplace=True)
        y_names = list(median_df[groupby])
    else:
        # 默认顺序：按原始数据中的出现顺序
        y_names = list(df[groupby].unique())

    # 2. 独立提取颜色映射
    # 无论 groupby 如何操作，我们直接从原始 df 中获取每个 group 对应的第一个颜色
    colors = []

    if "color" in df_columns:
        # 获取去重后的 Group-Color 对应关系 (保留首次出现的颜色)
        unique_color_df = df.drop_duplicates(subset=[groupby], keep='first')
        # 构建映射字典: Group Name -> Color Hex
        color_map = dict(zip(unique_color_df[groupby], unique_color_df["color"]))

        # 根据 y_names 的顺序构建 colors 列表
        # 使用 get 方法，如果找不到对应颜色则填充 None (或默认色)
        colors = [color_map.get(name) for name in y_names]

    # Construct kwargs for catplot
    plot_kwargs = kwargs.copy()

    if kind == "violin":
        plot_kwargs['split'] = split

    effective_palette = palette if palette is not None else (colors if "color" in df_columns else None)

    # scatter (categorical plot)
    g = sns.catplot(
        data=df,
        x=groupby,
        y=value,
        kind=kind,
        hue=hue,
        order=y_names,
        linewidth=line_width,
        palette=effective_palette,
        **plot_kwargs
    )

    # set coordinate style
    for _ax_ in g.axes.flat:
        _ax_.spines['top'].set_linewidth(line_width)
        _ax_.spines['right'].set_linewidth(line_width)
        _ax_.spines['bottom'].set_linewidth(line_width)
        _ax_.spines['left'].set_linewidth(line_width)
        # Set the rotation angle of the x-axis labels
        _ax_.tick_params(axis='x', rotation=rotation)

    plot_end(title, x_name, y_name, output, show, close)

    return g


def violin_significance(
    df: DataFrame,
    value: str,
    groupby: str,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    palette: Union[Tuple, list, str, dict] = None,
    hue: Optional[str] = None,
    order_names: Optional[list] = None,
    pairs: List[tuple] = None,
    test: test_method_type = "Mann-Whitney",
    rotation: float = 65,
    marker_size: float = 1,
    strip_alpha: float = 0.8,
    line_width: float = 0.5,
    strip_line_width: float = 0.2,
    strip_jitter: Union[float, Literal[True]] = True,
    text_line_width: float = 0.5,
    text_font_size: float = 6,
    top_offset: float = 0,
    title: Optional[str] = None,
    is_show_test: bool = True,
    test_format: Literal["text", "symbol"] = "symbol",
    output: Optional[path] = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Violin plot + stripplot + significance bars.
    显著性横线端点直接从绘制结果中反推真实小提琴中心,不依赖 dodge 公式。
    """

    def _p_to_text(p_val_: float) -> str:
        if p_val_ < 1e-4:
            return f'P={p_val_:.2e}' if test_format == "text" else "****"
        elif p_val_ < 1e-3:
            return f'P={p_val_:.2e}' if test_format == "text" else "***"
        elif p_val_ < 1e-2:
            return f'P={p_val_:.3e}' if test_format == "text" else "**"
        elif p_val_ < 0.05:
            return f'P={p_val_:.3f}' if test_format == "text" else "*"
        else:
            return f'P={p_val_:.2f}' if test_format == "text" else "ns"

    def _extract_violin_centers(ax_) -> list:
        """从绘制结果提取每个小提琴体的真实 x 中心(每个 Path 对应一个半小提琴)。"""
        centers = []
        for coll in ax_.collections:
            # PolyCollection 只会来自 violinplot; stripplot 的是 PathCollection,不会被误捕
            if isinstance(coll, PolyCollection):
                for p in coll.get_paths():
                    centers.append(float(p.vertices[:, 0].mean()))
        return centers

    def _build_center_lookup(ax_) -> dict:
        """
        构建 (group, hue) -> 真实 x 中心 的查找表。
        步骤:
          1. 每个小提琴中心归属到最近的 x 刻度 -> group;
          2. 组内按中心 x 排序,与该组"实际有数据的 hue 水平"(按 seaborn 左->右顺序)一一对应。
        """
        centers = _extract_violin_centers(ax_)
        ticks = np.arange(len(y_names))

        by_group: dict = {g: [] for g in y_names}
        for c in centers:
            gi = int(np.argmin(np.abs(ticks - c)))
            by_group[y_names[gi]].append(c)

        lookup: dict = {}

        if hue is None:
            # 无 hue:每组最多一个小提琴
            for g in y_names:
                if len(by_group[g]) == 1:
                    lookup[(g, None)] = by_group[g][0]
            return lookup

        # 有 hue:组内按 hue 左->右顺序对应
        hue_order_vals = kwargs.get("hue_order", None) or list(df[hue].unique())
        for g in y_names:
            cs = sorted(by_group[g])
            hues_present = [
                h for h in hue_order_vals
                if df.loc[(df[groupby] == g) & (df[hue] == h), value].notna().any()
            ]
            # 数量对得上才做对应,避免错位
            if cs and len(cs) == len(hues_present):
                for h, c in zip(hues_present, cs):
                    lookup[(g, h)] = c
        return lookup

    # --- 1. Parameter Validation ---
    df_columns = list(df.columns)

    if value not in df_columns:
        log.error(f"The `value` ({value}) parameter must be in `df` columns ({df_columns})")
        raise ValueError(f"The `value` ({value}) parameter must be in `df` columns ({df_columns})")

    if hue is not None and hue not in df_columns:
        log.error(f"The `hue` ({hue}) parameter must be in `df` columns ({df_columns})")
        raise ValueError(f"The `hue` ({hue}) parameter must be in `df` columns ({df_columns})")

    # --- 2. Initialize Canvas ---
    fig, ax = plot_start()

    # --- 3. Handling Order and Colors ---
    new_df: DataFrame = df.groupby([groupby], as_index=False)[value].median()

    colors: list = []

    if "color" in df_columns:
        new_df_color: DataFrame = df.groupby([groupby], as_index=False)["color"].first()
        new_df = new_df.merge(new_df_color, how="left", on=groupby)

    if order_names is None:
        new_df.sort_values([value], ascending=False, inplace=True)
        y_names: list = list(new_df[groupby])

        if "color" in df_columns:
            colors = list(new_df["color"])
    else:
        y_names = order_names

        if "color" in df_columns:
            new_df.index = new_df[groupby]

            for i in order_names:
                if i in new_df.index:
                    c_val = new_df.loc[i, "color"]
                    colors.append(c_val.iloc[0] if isinstance(c_val, (list, np.ndarray, pd.Series)) else c_val)

    final_palette = palette if palette is not None else (colors if len(colors) > 0 else "Set2")

    # 允许用户覆盖 dodge,同时保证判断与实际绘制一致
    do_dodge = kwargs.pop("dodge", True if hue else False)

    # --- 4. Plot violin & strip ---
    sns.violinplot(
        data=df,
        x=groupby,
        y=value,
        hue=hue,
        order=y_names,
        palette=final_palette,
        linewidth=line_width,
        ax=ax,
        dodge=do_dodge,
        **kwargs
    )

    # !!! 关键:在 stripplot 之前立即捕获小提琴真实中心
    center_lookup: dict = _build_center_lookup(ax)

    sns.stripplot(
        data=df,
        x=groupby,
        y=value,
        hue=hue,
        order=y_names,
        palette=final_palette,
        size=marker_size,
        edgecolor='black',
        linewidth=strip_line_width,
        alpha=strip_alpha,
        jitter=strip_jitter,
        ax=ax,
        dodge=do_dodge
    )

    # --- 5. Significance Annotation ---
    if is_show_test:
        y_min, y_max = df[value].min(), df[value].max()
        y_range = (y_max - y_min) or 1.0

        if hue:
            hue_order_vals = kwargs.get("hue_order", None) or list(df[hue].unique())
            if pairs is None:
                pairs = [(hue_order_vals[i], hue_order_vals[i + 1])
                         for i in range(len(hue_order_vals) - 1)]
        else:
            if pairs is None:
                pairs = [(y_names[i], y_names[i + 1]) for i in range(len(y_names) - 1)]

        level_counter: dict = {}

        for hue_val1, hue_val2 in pairs:
            for group_name in y_names:

                # 取数据
                if hue:
                    data1 = df.loc[(df[groupby] == group_name) & (df[hue] == hue_val1), value].dropna()
                    data2 = df.loc[(df[groupby] == group_name) & (df[hue] == hue_val2), value].dropna()
                else:
                    data1 = df.loc[df[groupby] == hue_val1, value].dropna()
                    data2 = df.loc[df[groupby] == hue_val2, value].dropna()

                if data1.empty or data2.empty:
                    continue

                # x 位置:直接查真实中心表,查不到(该组合没画出来)就跳过
                x_pos1 = center_lookup.get((group_name, hue_val1) if hue else (hue_val1, None))
                x_pos2 = center_lookup.get((group_name, hue_val2) if hue else (hue_val2, None))
                if x_pos1 is None or x_pos2 is None:
                    continue

                p_val = get_stat_result(data1, data2, test).pvalue

                # y 位置:基于该比较对的局部最大值,按 group 独立递增
                level = level_counter.get(group_name, 0) + 1
                level_counter[group_name] = level
                curr_h = max(data1.max(), data2.max()) + (0.05 + 0.08 * (level - 1)) * y_range

                ax.plot([x_pos1, x_pos1, x_pos2, x_pos2],
                        [curr_h, curr_h + 0.02 * y_range,
                         curr_h + 0.02 * y_range, curr_h],
                        lw=text_line_width, color='black')

                ax.text((x_pos1 + x_pos2) * 0.5, curr_h + 0.025 * y_range,
                        _p_to_text(p_val),
                        ha='center', va='bottom', fontsize=text_font_size)

    # --- 6. Aesthetics Adjustments ---
    for spine in ax.spines.values():
        spine.set_linewidth(line_width)

    ax.spines['top'].set_position(('outward', top_offset))
    ax.tick_params(axis='x', rotation=rotation)

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        n_hues = len(df[hue].unique())
        ax.legend(handles[:n_hues], labels[:n_hues], title=hue)

    plot_end(title, x_name, y_name, output, show, close)

    return ax

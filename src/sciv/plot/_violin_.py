# -*- coding: UTF-8 -*-

from typing import Tuple, Union, Literal, Any, List

import numpy as np
import pandas as pd
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
    value: str = "value",
    x_name: str = None,
    y_name: str = "value",
    groupby: str = "clusters",
    palette: Union[Tuple, list, str] = None,
    hue: str = None,
    order_names: list = None,
    pairs: List[tuple] = None,
    test: test_method_type = "Mann-Whitney",
    rotation: float = 65,
    marker_size: float = 2,
    line_width: float = 0.5,
    title: str = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):
    """
    Plot violin plot with scatter overlay and significance bars.

    Combines violin (distribution), boxplot (quartiles), and stripplot (raw data).
    Performs statistical test and annotates p-values.

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
    groupby : str, optional
        Clusters column.
    palette : Union[Tuple, list], optional
        Palette.
    hue : str, optional
        Hue column.
    order_names : list, optional
        Order names.
    pairs : List[tuple], optional
        List of tuple pairs to compare. If None, compares all sequential pairs.
    test : str, default "t-test_ind"
        Statistical test for pairwise comparisons. Options include:
        {"t-test_ind", "t-test_welch", "t-test_paired", "Mann-Whitney", "Mann-Whitney-gt",
         "Mann-Whitney-ls", "Levene", "Wilcoxon", "Kruskal", "Brunner-Munzel"}.
    rotation : float, optional
        Rotation.
    marker_size : float, optional
        Line width.
    line_width : float, optional
        Line width.
    title : str, optional
        Title.
    output : path, optional
        Output path.
    show : bool, optional
        Whether to show.
    close : bool, optional
        Whether to close.
    kwargs : Any, optional
        Keyword arguments passed to sns.violinplot.
    """

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
    group_columns = [groupby]
    new_df: DataFrame = df.groupby(group_columns, as_index=False)[value].median()

    colors: list = []
    if "color" in df_columns:
        new_df_color: DataFrame = df.groupby(group_columns, as_index=False)["color"].first()
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

    final_palette = palette if palette is not None else (colors if len(colors) > 0 else None)

    # --- 4. Plotting Layers ---

    # Layer 1: Violin (Distribution)
    sns.violinplot(
        data=df,
        x=groupby,
        y=value,
        hue=hue,
        order=y_names,
        inner='box',
        palette=final_palette,
        linewidth=line_width,
        ax=ax,
        dodge=True if hue else False,
        **kwargs
    )

    # Layer 3: Stripplot (Raw Data)
    sns.stripplot(
        data=df,
        x=groupby,
        y=value,
        hue=hue,
        order=y_names,
        palette=final_palette,
        size=marker_size,
        edgecolor='black',
        linewidth=line_width,
        alpha=0.6,
        jitter=True,
        ax=ax,
        dodge=True if hue else False
    )

    # --- 5. Significance Annotation ---
    if hue:
        if pairs is None:
            hue_values = df[hue].unique()
            pairs = [(hue_values[i], hue_values[i + 1]) for i in range(len(hue_values) - 1)]

        y_max = df[value].max()
        y_range = df[value].max() - df[value].min()
        # Start annotation slightly above the max data point
        h = y_max + 0.05 * y_range

        # x_tick_map = {name: i for i, name in enumerate(y_names)}  # 不再需要这个

        # 只定义一次 hue_values 和 hue_positions
        hue_values = df[hue].unique()
        hue_positions = {hue_val: i for i, hue_val in enumerate(hue_values)}

        for i, (hue_val1, hue_val2) in enumerate(pairs):

            for group_name in y_names:
                data1 = df[(df[groupby] == group_name) & (df[hue] == hue_val1)][value]
                data2 = df[(df[groupby] == group_name) & (df[hue] == hue_val2)][value]

                if len(data1) == 0 or len(data2) == 0:
                    continue

                # Calculate P-value
                p_val = get_stat_result(data1, data2, test).pvalue

                # Draw connectors - 现在在同一个 group 内部比较
                x_pos = y_names.index(group_name)  # 获取 group 的位置

                # 计算 hue 在 group 内部的相对位置
                x_pos1 = x_pos + (hue_positions[hue_val1] - (len(hue_values) - 1) / 2) / (len(hue_values) + 1)
                x_pos2 = x_pos + (hue_positions[hue_val2] - (len(hue_values) - 1) / 2) / (len(hue_values) + 1)

                level = i + 1
                curr_h = h + (0.1 * y_range * (level - 1))

                ax.plot([x_pos1, x_pos1, x_pos2, x_pos2],
                        [curr_h, curr_h + 0.02 * y_range, curr_h + 0.02 * y_range, curr_h],
                        lw=1.5, color='black')

                # Format P-value text
                if np.isnan(p_val):
                    p_text = "NaN"
                elif p_val < 0.001:
                    p_text = f'P={p_val:.2e}'
                elif p_val < 0.05:
                    p_text = f'P={p_val:.3f}'
                else:
                    p_text = f'P={p_val:.2f}'

                ax.text((x_pos1 + x_pos2) * 0.5, curr_h + 0.025 * y_range, p_text,
                        ha='center', va='bottom')

    # --- 6. Aesthetics Adjustments ---
    ax.spines['top'].set_linewidth(line_width)
    ax.spines['right'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.tick_params(axis='x', rotation=rotation)

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        n_hues = len(df[hue].unique())
        ax.legend(handles[:n_hues], labels[:n_hues], title=hue)

    plot_end(title, x_name, y_name, output, show, close)

    return ax

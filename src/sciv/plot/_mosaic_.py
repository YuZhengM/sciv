# -*- coding: UTF-8 -*-

from typing import Optional, Union, Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency

from .. import util as ul
from ..util import path, plot_end, plot_start, matrix_data

__name__: str = "plot_line"

log = ul.log(__name__, "ERROR")


def odds_ratio(
    data: Union[list, matrix_data],
    row_labels: Union[list, tuple[str, str]] = ('Anchor', 'HD'),
    col_labels: Union[list, tuple[str, str]] = ('Present', 'Absent'),
    title: Optional[str] = None,
    colors: Union[list, tuple[str, str]] = ('#4878B0', '#F2F2F2'),
    font_size: float = 5,
    gap_value: float = 0.015,
    output: Optional[path] = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
):

    data = np.array(data)

    if data.shape != (2, 2):
        log.error(f"Input data has shape {data.shape}, but a 2x2 contingency table is required for odds ratio calculation.")
        raise ValueError(f"Expected a 2x2 contingency table, got data with shape {data.shape}.")
   
    # 统计检验
    chi2, p, dof, _ = chi2_contingency(data, correction=False)
    a, b, c, d = data.ravel()
    or_value = a * d / (b * c)
    se_value = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_low, ci_high = np.exp(np.log(or_value) - 1.96 * se_value), np.exp(np.log(or_value) + 1.96 * se_value)

    def _fmt_p_():

        if p < 1e-300:
            return "P < 1.0 × 10$^{-300}$"

        exp = int(np.floor(np.log10(p)))
        mant = p / 10 ** exp
        return rf"P = {mant:.1f} × 10$^{{{exp}}}$"

    # mosaic 布局
    row_fracs = data.sum(axis=1) / data.sum()
    col_fracs = data / data.sum(axis=1, keepdims=True)

    fig, ax = plot_start()

    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("linewidth", 0.5)

    y_top = 1.0

    for i, (rl, rf) in enumerate(zip(row_labels, row_fracs)):
        y_bottom = y_top - rf
        x_left = 0.0

        for j, (cl, cf) in enumerate(zip(col_labels, col_fracs[i])):
            w = cf
            color = colors[0] if j == 0 else colors[1]
            ax.add_patch(plt.Rectangle(
                (x_left, y_bottom), w, rf - gap_value * (i < len(row_fracs) - 1), facecolor=color, **kwargs)
            )
            ax.text(
                x_left + w / 2, y_bottom + (rf - gap_value) / 2, f'{data[i, j]:,}',
                ha='center', va='center', fontsize=font_size, color='black'
            )

            if i == 0:
                ax.text(x_left + w / 2, y_top + 0.01, cl, ha='center', va='bottom', fontsize=font_size)

            x_left += w

        ax.text(-0.05, y_bottom + rf / 2, rl, ha='right', va='center', fontsize=font_size)
        y_top = y_bottom - gap_value

    # 标题与统计文字
    ax.text(-0.05, 1.16, title, fontsize=font_size + 1, ha='left', va='top')
    stats_txt = (f"Odds ratio = {or_value:.2f}, 95% CI ({ci_low:.2f}-{ci_high:.2f})\n"
                 f"Pearson's chi-squared test\n"
                 f"{_fmt_p_()}")

    ax.text(-0.05, -0.08, stats_txt, fontsize=font_size, ha='left', va='top')

    ax.set_aspect('equal')
    ax.set_xlim(-0.25, 1.05)
    ax.set_ylim(-0.25, 1.20)
    ax.axis('off')

    plot_end(title, None, None, output, show, close)

    return ax

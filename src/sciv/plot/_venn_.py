# -*- coding: UTF-8 -*-

from typing import Any

from matplotlib.figure import Figure
from matplotlib_venn import venn3, venn3_circles, venn2, venn2_circles

from .. import util as ul
from ..util import path, collection, type_set_colors, plot_end, plot_start

__name__: str = "plot_venn"

log = ul.log(__name__, "ERROR")


def venn_three(
    set1: collection,
    set2: collection,
    set3: collection,
    name1: str = "Set1",
    name2: str = "Set2",
    name3: str = "Set3",
    colors: list = None,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Figure, Any]:
    """
    Plot three Venn diagram.

    Parameters
    ----------
    set1 : collection
        First set of elements.
    set2 : collection
        Second set of elements.
    set3 : collection
        Third set of elements.
    name1 : str, optional
        Name of the first set.
    name2 : str, optional
        Name of the second set.
    name3 : str, optional
        Name of the third set.
    colors : list, optional
        Colors for the sets.
    x_name : str, optional
        X name.
    y_name : str, optional
        Y name.
    title : str, optional
        Title of the diagram.
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
    fig, ax = plot_start()

    if colors is None:
        colors = type_set_colors[:3]

    if len(colors) < 3:
        log.info(f"The value of colors requires three elements.")
        raise ValueError(f"The value of colors requires three elements.")
    elif len(colors) > 3:
        colors = colors[:3]

    set1 = set(set1)
    set2 = set(set2)
    set3 = set(set3)

    subsets = (set1, set2, set3)

    venn3(subsets=subsets, set_labels=(name1, name2, name3), ax=ax, set_colors=colors, **kwargs)

    # noinspection PyTypeChecker
    venn3_circles(subsets=subsets, linestyle='dashed', linewidth=1, color="grey", ax=ax)

    ax.legend(loc='upper right')

    ax.axis('off')

    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, ax


def venn_two(
    set1: collection,
    set2: collection,
    name1: str = "Set1",
    name2: str = "Set2",
    colors: list = None,
    x_name: str = None,
    y_name: str = None,
    title: str = None,
    output: path = None,
    show: bool = False,
    close: bool = True,
    **kwargs: Any
) -> tuple[Figure, Any]:
    """
    Plot two Venn diagram.

    Parameters
    ----------
    set1 : collection
        First set of elements.
    set2 : collection
        Second set of elements.
    name1 : str, optional
        Name of the first set.
    name2 : str, optional
        Name of the second set.
    colors : list, optional
        Colors for the sets.
    x_name : str, optional
        X name.
    y_name : str, optional
        Y name.
    title : str, optional
        Title of the diagram.
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
    fig, ax = plot_start()

    if colors is None:
        colors = type_set_colors[:2]

    if len(colors) < 2:
        log.info(f"The value of colors requires three elements.")
        raise ValueError(f"The value of colors requires three elements.")
    elif len(colors) > 2:
        colors = colors[:2]

    set1 = set(set1)
    set2 = set(set2)

    venn2((set1, set2), set_labels=(name1, name2), ax=ax, set_colors=colors, **kwargs)

    # noinspection PyTypeChecker
    venn2_circles(subsets=(set1, set2), linestyle='dashed', linewidth=1, color="grey", ax=ax)

    ax.legend(loc='upper right')

    ax.axis('off')

    plot_end(fig, title, x_name, y_name, output, show, close)

    return fig, ax

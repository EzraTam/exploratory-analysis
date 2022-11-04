"""Visualization module
"""

from collections import Counter
from typing import Any, List, Tuple, Optional
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from exploratory_analysis import basic_functions as bf


def boxplot_list(
    data: pd.DataFrame, x: Any, list_y: List[Any], list_axes=List[Axes]
) -> None:
    for y in list_y:
        sns.boxplot(
            ax=list_axes[list_y.index(y)],
            data=data,
            x=x,
            y=y,
            showfliers=False,
        )


def piechart_list(
    li_data: List[pd.Series], labels: List[str], list_axes: List[Axes]
) -> None:

    colors = sns.color_palette("pastel")[0:5]

    for ax, data in zip(list_axes, li_data):
        ax.pie(
            x=data,
            labels=labels,
            colors=colors,
            explode=bf.set_zero_if_below(data / sum(data), 0.1),
            autopct="%.0f%%",
            pctdistance=1.1,
            labeldistance=1.2,
            shadow=True,
            radius=1,
            rotatelabels=True,
            textprops={"fontsize": 8},
        )


def plot_boxplots(
    data: pd.DataFrame,
    x_feature: str,
    li_y_features: List[str],
    title: str,
    figsize: Tuple[int, int],
) -> None:
    """Function for visualizing data by
    boxplot

    Args:
        data (pd.DataFrame): Input DF
        x_feature (str): Feature/column for the x-axis
            of the boxplot
        li_y_features (List[str]): List of features/columns
            to be plotted
        title (str): Title of the plot
        figsize (Tuple[int, int]): Size of the figure
    """
    assert (
        x_feature in data.columns
    ), "Data does not have the queried column for the x-axis"

    assert len(li_y_features) <= len(
        bf.distinct(li_y_features)
    ), "Elements in li_y_features has to be distinct"

    assert set(li_y_features).issubset(
        data.columns
    ), "Data does not have the queried column for the x-axis"

    fig, axes = plt.subplots(1, len(li_y_features), figsize=figsize)
    fig.suptitle(title)
    boxplot_list(data=data, x=x_feature, list_y=li_y_features, list_axes=axes)


def create_pie_chart_from_multi_idx_df(
    data: pd.DataFrame,
    title: str,
    col_plotted: str,
    ordering_sub_category: Optional[list] = None,
) -> None:
    """Create pie charts from two-level-indexed DF. Each pie chart corresponds
    to a first level index (a.k.a. category) and depicts the quantity specified
    in 'col_plotted' input for each second level index.
    Args:
        data (pd.DataFrame): DF with multilevel index
        title (str): Super title of the plot
        col_plotted (str): Column name of the quantity to be visualize
        ordering_sub_category (default=[]): List of second level indexes ordered
            desirely. Optional!
    """

    bf.raise_error(
        (data.index.nlevels != 2), ValueError, "DF has to have 2 level indexes"
    )

    categories = list(set(data.index.get_level_values(0)))
    sub_categories = list(set(data.index.get_level_values(1)))

    bf.raise_error(
        condition=(ordering_sub_category is not None)
        and (Counter(ordering_sub_category) != Counter(sub_categories)),
        error_class=ValueError,
        msg="ordering_sub_category has to be a permutation of the second level indexes",
    )

    fig, axes = plt.subplots(1, len(categories), figsize=(25, 8))
    fig.suptitle(title)

    li_data_raw = [data.xs(category)[col_plotted] for category in categories]

    data_processed = [
        bf.return_obj_val_in_list_order(data_raw, ordering_sub_category)
        if ordering_sub_category
        else data_raw
        for data_raw in li_data_raw
    ]

    labels = ordering_sub_category if ordering_sub_category else sub_categories

    piechart_list(li_data=data_processed, labels=labels, list_axes=axes)

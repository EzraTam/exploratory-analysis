import pandas as pd
import seaborn as sns
from typing import List, Tuple,Optional
import matplotlib.pyplot as plt
from collections import Counter


def plot_boxplots(
    data: pd.DataFrame,
    x_feature: str,
    li_y_features: List[str],
    title: str,
    figsize: Tuple[int, int],
) -> None:
    assert (
        x_feature in data.columns
    ), "Data does not have the queried column for the x-axis"
    for y_feature in li_y_features:
        assert (
            y_feature in data.columns
        ), "Data does not have the queried column for the x-axis"
    fig, axes = plt.subplots(1, len(li_y_features), figsize=figsize)
    fig.suptitle(title)
    for y_feature in li_y_features:
        sns.boxplot(
            ax=axes[li_y_features.index(y_feature)],
            data=data,
            x=x_feature,
            y=y_feature,
            showfliers=False,
        )


def create_pie_chart_from_multi_idx_df(
    data: pd.DataFrame, title: str, col_plotted: str, ordering_sub_category: Optional[list] = None
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

    if data.index.nlevels != 2:
        raise ValueError("DF has to have 2 level indexes")

    fg_order_subcategory = False
    if ordering_sub_category is None:
        fg_order_subcategory = True
        if Counter(ordering_sub_category) != Counter(
            list(set(data.index.get_level_values(1)))
        ):
            raise ValueError(
                "ordering_sub_category has to be a permutation of the second level indexes"
            )

    categories = list(set(data.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(categories), figsize=(25, 8))
    fig.suptitle(title)
    for category in categories:
        data_raw = data.xs(category)[col_plotted]

        if fg_order_subcategory:
            data_plot = [
                data_raw[sub_category] for sub_category in ordering_sub_category
            ]
            labels = ordering_sub_category
        else:
            data_plot = data_raw
            labels = list(set(data.index.get_level_values(1)))

        explode = [val if val >= 0.1 else 0 for val in data_plot / sum(data_plot)]

        # define Seaborn color palette to use
        colors = sns.color_palette("pastel")[0:5]

        ax_idx = categories.index(category)

        axes[ax_idx].set_title(category, pad=40)

        # create pie chart
        axes[ax_idx].pie(
            x=data_plot,
            labels=labels,
            colors=colors,
            explode=explode,
            autopct="%.0f%%",
            pctdistance=1.1,
            labeldistance=1.2,
            shadow=True,
            radius=1,
            rotatelabels=True,
            textprops={"fontsize": 8},
        )

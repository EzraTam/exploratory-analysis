"""Visualization module
"""

from collections import Counter
from typing import Any, List, Tuple, Optional, Union
from matplotlib.axes import Axes
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import pandas as pd
import seaborn as sns

from exploratory_analysis import basic_functions as bf


def boxplot_list(
    data: pd.DataFrame, x: Any, list_y: List[Any], list_axes=List[Axes]
) -> None:
    """Plot boxplots from a list

    Args:
        data (pd.DataFrame): _description_
        x (Any): _description_
        list_y (List[Any]): _description_
        list_axes (_type_, optional): _description_. Defaults to List[Axes].
    """
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
    """Plot piechart from a list

    Args:
        li_data (List[pd.Series]): _description_
        labels (List[str]): _description_
        list_axes (List[Axes]): _description_
    """

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


class MultiIdxDF:
    """Class for handling multi index DF"""

    def __init__(self, multi_idx_df: pd.DataFrame):
        bf.raise_error(
            (multi_idx_df.index.nlevels != 2),
            ValueError,
            "DF has to have 2 level indexes",
        )
        self.multi_idx_df = multi_idx_df

    @property
    def categories(self) -> List[str]:
        """List of keys in the first level index
        of DF
        """
        return bf.distinct(self.multi_idx_df.index.get_level_values(0))

    @property
    def sub_categories(self) -> List[str]:
        """List of keys in the second level index
        of DF
        """
        return bf.distinct(self.multi_idx_df.index.get_level_values(1))

    def _separate_multi_idx_df(self, which_col: str) -> List[pd.Series]:
        """Separate multi index DF in series respective to first level index

        Args:
            which_col (str): Column to extract for each series
        """
        return [
            self.multi_idx_df.xs(category)[which_col] for category in self.categories
        ]

    def create_pie_chart(
        self,
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
            condition=(ordering_sub_category is not None)
            and (Counter(ordering_sub_category) != Counter(self.sub_categories)),
            error_class=ValueError,
            msg="ordering_sub_category has to be a permutation of the second level indexes",
        )

        fig, axes = plt.subplots(1, len(self.categories), figsize=(25, 8))
        fig.suptitle(title)

        data_processed = [
            bf.return_obj_val_in_list_order(data_raw, ordering_sub_category)
            if ordering_sub_category
            else data_raw
            for data_raw in self._separate_multi_idx_df(which_col=col_plotted)
        ]

        labels = ordering_sub_category if ordering_sub_category else self.sub_categories

        piechart_list(li_data=data_processed, labels=labels, list_axes=axes)


def create_pie_chart_from_multi_idx_df(
    multi_idx_df: pd.DataFrame,
    title: str,
    col_plotted: str,
    ordering_sub_category: Optional[list] = None,
) -> None:
    """Function for creating pie charts from multi index DF
    respective to first level index

    Args:
        multi_idx_df (pd.DataFrame): _description_
        title (str): _description_
        col_plotted (str): _description_
        ordering_sub_category (Optional[list], optional): _description_. Defaults to None.
    """

    MultiIdxDF(multi_idx_df=multi_idx_df).create_pie_chart(
        title=title,
        col_plotted=col_plotted,
        ordering_sub_category=ordering_sub_category,
    )


def add_median_labels(ax: Axes, fmt=".1f"):
    """Function to add median labels to a boxplot

    Args:
        ax (Axes): Axes with Boxplot
        fmt (str, optional): Format of the displayed number. Defaults to '.1f'.
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4 : len(lines) : lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(
            x,
            y,
            f"{value:{fmt}}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )


def plot_box_count(
    df_to_visualize: pd.DataFrame,
    x_category: str,
    y_quantity: str,
    legend_category: str,
    stat_xlabel: Optional[str] = None,
    stat_ylabel: Optional[str] = None,
    sup_title: Optional[str] = None,
):
    """Function to plot the boxplot of data and the corresponding count

    Args:
        df_to_visualize (pd.DataFrame): DF to be analyzed
        x_category (str): Category of the boxes on the x-axis
        y_quantity (str): Quantity to be analyzed - specify the y-axis of the data
        legend_category (str): Category for the legend a.k.a hue
        stat_xlabel (Optional[str], optional): Label of the x-axis. Defaults to None.
        stat_ylabel (Optional[str], optional): Label of the y-axis. Defaults to None.
        sup_title (Optional[str], optional): Title of the plot. Defaults to None.
    """

    # Create subaxes
    fig, axs = plt.subplots(
        nrows=2, gridspec_kw={"height_ratios": [3, 0.5]}, figsize=(30, 15)
    )

    if sup_title:
        fig.suptitle(sup_title, fontsize=25, fontweight="bold")

    # Configuration for mean visualization
    mean = {
        "showmeans": True,
        "meanprops": {
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "4",
        },
    }

    box_plot = sns.boxplot(
        data=df_to_visualize,
        x=x_category,
        y=y_quantity,
        hue=legend_category,
        showfliers=False,
        **mean,
        ax=axs[0],
    )

    add_median_labels(box_plot)

    if stat_xlabel is not None:
        box_plot.set_xlabel(stat_xlabel)

    if stat_ylabel is not None:
        box_plot.set_ylabel(stat_ylabel)

    df_count = (
        df_to_visualize.groupby([legend_category, x_category], as_index=False)[
            y_quantity
        ]
        .count()
        .rename(columns={y_quantity: "Number Data"})
    )
    count_plot = sns.barplot(
        data=df_count, x=x_category, y="Number Data", hue=legend_category, ax=axs[1]
    )

    # Add values to the bars
    for i in count_plot.containers:
        count_plot.bar_label(
            i,
        )

    # Delete legend - Need a more elegant solution, e.g. create legend for both plots
    count_plot.get_legend().remove()

    plt.show()


def plot_heat_map_from_matrices(
    dfs_matrix: List[Tuple[Union[str, int], pd.DataFrame]], plot_title: str
) -> None:
    """Given matrices, plot multiple heat maps

    Args:
        dfs_matrix (List[Tuple[Union[str,int],pd.DataFrame]]): Inpot aggregation matrices
        plot_title (str): Title of the plot
    """

    # Has to be adjustable later
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["r", "y", "g", "g", "g", "g"], N=256
    )

    fig, axs = plt.subplots(nrows=len(dfs_matrix), figsize=(30, 15), sharex="row")
    fig.suptitle(plot_title, fontsize=25)

    config_heatmap = {
        "cmap": cmap,
        "annot": True,
        "linewidth": 0.5,
        "vmin": 0,
        "vmax": 4,
        "square": True,
        "fmt": ".2f",
    }
    font_config = {"fontweight": "bold", "fontsize": 20, "pad": 15}

    for idx, (_cat_matrix, _df_matrix) in enumerate(dfs_matrix):
        sns.heatmap(_df_matrix, ax=axs[idx], **config_heatmap)
        axs[idx].set_title(_cat_matrix, **font_config)

    fig.subplots_adjust(wspace=0.01)

    plt.show()

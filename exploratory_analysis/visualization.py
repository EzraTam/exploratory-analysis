"""Visualization module
"""

from collections import Counter
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.cm as cmx
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from exploratory_analysis import basic_functions as bf
from exploratory_analysis.utils import fill_df_full_cat


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

    colors_from_palette = sns.color_palette("pastel")[0:5]

    for ax, data in zip(list_axes, li_data):
        ax.pie(
            x=data,
            labels=labels,
            colors=colors_from_palette,
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
    xlabel: Optional[str] = None,
    stat_ylabel: Optional[str] = None,
    sup_title: Optional[str] = None,
    legend_title: Optional[str] = None,
    x_category_full: Optional[List[Union[str, int, float]]] = None,
    legend_category_full: Optional[List[Union[str, int, float]]] = None,
    figsize=(30, 15)
):
    """Function to plot the boxplot of data and the corresponding count

    Args:
        df_to_visualize (pd.DataFrame): DF to be analyzed
        x_category (str): Category of the boxes on the x-axis
        y_quantity (str): Quantity to be analyzed - specify the y-axis of the data
        legend_category (str): Category for the legend a.k.a hue
        xlabel (Optional[str], optional): Label of the x-axis. Defaults to None.
        stat_ylabel (Optional[str], optional): Label of the y-axis. Defaults to None.
        sup_title (Optional[str], optional): Title of the plot. Defaults to None.
        legend_title (Optional[str], optional): Title of the legend. Defaults to None.
        x_category_full (Optional[List[Union[str,int,float]]], optional): Full list of category realizations of the x-axis. Defaults to None.
        legend_category_full (Optional[List[Union[str,int,float]]], optional): Full list of category realizations of the legend. Defaults to None.
    """
    # Create subaxes
    fig, axs = plt.subplots(
        nrows=2, gridspec_kw={"height_ratios": [3, 0.5]}, figsize=figsize
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

    df_count = (
        df_to_visualize.groupby([legend_category, x_category], as_index=False)[
            y_quantity
        ]
        .count()
        .rename(columns={y_quantity: "Number Data"})
    )

    if x_category_full is None:
        x_category_full = list(df_count[x_category].unique())
        x_category_full.sort()
    if legend_category_full is None:
        legend_category_full = list(df_count[legend_category].unique())
        legend_category_full.sort()
    if legend_title is None:
        legend_title = str(legend_category)

    _cats = {
        x_category: x_category_full,
        legend_category: legend_category_full,
    }

    df_count = fill_df_full_cat(df_count, _cats)

    box_plot = sns.boxplot(
        data=df_to_visualize.rename(columns={legend_category: legend_title}),
        x=x_category,
        y=y_quantity,
        hue=legend_title,
        order=x_category_full,
        hue_order=legend_category_full,
        legend="full",
        showfliers=False,
        **mean,
        ax=axs[0],
    )

    add_median_labels(box_plot)

    if xlabel is not None:
        box_plot.set_xlabel(xlabel)

    if stat_ylabel is not None:
        box_plot.set_ylabel(stat_ylabel)

    count_plot = sns.barplot(
        data=df_count, x=x_category, y="Number Data", hue=legend_category, ax=axs[1]
    )

    # Add values to the bars
    for i in count_plot.containers:
        count_plot.bar_label(
            i,
        )
    count_plot.set_xlabel(xlabel)

    # Delete legend - Need a more elegant solution, e.g. create legend for both plots
    count_plot.get_legend().remove()

    return axs


def plot_heat_map_from_matrices(
    dfs_matrix: List[Tuple[Union[str, int], pd.DataFrame]],
    plot_title: str,
    xlabel: Optional[str] = None,
    xticks_labels: Optional[List[Union[str,int]]] = None,
    ylabel: Optional[str] = None,
    vmin: Optional[float] = 0,
    vmax: Optional[float] = 4,
    figsize: Optional[Tuple[int]] = (30,15),
    prop_color: Optional[List[str]] = ["r", "r", "y", "y", "g", "g"]
) -> None:
    """Given matrices, plot multiple heat maps

    Args:
        dfs_matrix (List[Tuple[Union[str,int],pd.DataFrame]]): Inpot aggregation matrices
        plot_title (str): Title of the plot
    """

    # Has to be adjustable later
    cmap = LinearSegmentedColormap.from_list(
        "rg", prop_color, N=256
    )

    fig, axs = plt.subplots(nrows=len(dfs_matrix), figsize=figsize, sharex="row")
    fig.suptitle(plot_title, fontsize=25)

    config_heatmap = {
        "cmap": cmap,
        "annot": True,
        "linewidth": 0.5,
        "vmin": vmin,
        "vmax": vmax,
        "square": True,
        "fmt": ".2f",
    }
    font_config = {"fontweight": "bold", "fontsize": 20, "pad": 15}

    if len(dfs_matrix) == 1:
        axs = [axs]

    for idx, (_cat_matrix, _df_matrix) in enumerate(dfs_matrix):
        sns.heatmap(_df_matrix, ax=axs[idx], **config_heatmap)
        axs[idx].set_title(_cat_matrix, **font_config)
        if xlabel is not None:
            axs[idx].set_xlabel(xlabel)
        if ylabel is not None:
            axs[idx].set_ylabel(ylabel)
        if xticks_labels is not None:
            axs[idx].set_xticklabels(xticks_labels)

    fig.subplots_adjust(hspace=0.5)

    plt.show()


## Sankey Diagram


def create_color_from_values(values: List[float], color_palette: str) -> List[float]:
    """Create colors from number of values

    Args:
        values (List[float]): List of values to transform
        color_palette (str): Color palette name, e.g. 'RdYlGn' and 'Spectral'

    Returns:
        List[float]: List of colors
    """

    norm = colors.Normalize(vmin=min(values), vmax=max(values))

    cmap = cmx.get_cmap(color_palette)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

    _result = map(scalar_map.to_rgba, values)

    return list(map(colors.to_hex, _result))


def _extract_source_target_label(
    df: pd.DataFrame, source_target: Tuple[str, str]
) -> Dict[str, List[Union[str, int]]]:
    """Extract the source and target labels from a DF

    Args:
        df (pd.DataFrame): DF input
        source_target (Tuple[str,str]): (source column name, target column name)

    Returns:
        Dict[str,List[Union[str,int]]]:
    """
    return {
        source_target[0]: list(df[source_target[0]].unique()),
        source_target[1]: list(df[source_target[1]].unique()),
    }


def _create_colors_from_data(
    data_tuple: Tuple[Any, Any, Union[float, int]],
    color_palette: Optional[str] = "RdYlGn",
) -> List[str]:
    """From data of the form (..., ..., value) create colors from the values

    Args:
        data_tuple (Tuple[Any,Any,Union[float,int]]): Data in form of list of tuples
        color_palette (Optional[str], optional): Color palettes. Defaults to "RdYlGn".

    Returns:
        List[str]: List of color values in hex
    """

    # Values is in the third entry in each of the tuple
    _colors = map(lambda x: bf.extract_from_tuple(x, 2), data_tuple)

    # Assign colors to the values - for each sorce target level one crreation
    # consider to extend
    return list(map(lambda x: create_color_from_values(x, color_palette), _colors))


def transform_data_to_source_target(
    dfs: Iterable[pd.DataFrame], source_target_pairs: Iterable[Tuple[str, str]]
):
    """Create source target data with color on values"""

    # Extract data from dfs
    _data = list(map(bf.df_to_list, dfs))

    _colors = _create_colors_from_data(data_tuple=_data)

    # # Values is in the third entry in each of the tuple
    # _colors = map(lambda x: extract_from_tuple(x,2), _data)

    # # Assign colors to the values - for each sorce target level one crreation
    # # consider to extend
    # _colors = map(lambda x: create_color_from_values(x,"RdYlGn"),_colors)

    # Unpack the data to a long vector and color
    data_coloured = zip(*map(lambda x: list(chain.from_iterable(x)), [_data, _colors]))
    data_coloured = [(*element[0], element[1]) for element in data_coloured]

    ## Create label for the nodes
    _dfs_cat = zip(dfs, source_target_pairs)

    _source_target_labels = map(
        lambda _tuple: _extract_source_target_label(_tuple[0], _tuple[1]), _dfs_cat
    )

    # Merging dict of labels and collect values on same keys into a list
    _source_target_labels = bf.merge_dicts_into_list_values(_source_target_labels)

    # drop duplicates in each of the lists
    # _source_target_labels = map(lambda x: list(set(x)), _source_target_labels.values())
    label_categorized = {
        key: list(set(value)) for key, value in _source_target_labels.items()
    }

    return data_coloured, label_categorized


def df_to_dag(
    df: pd.DataFrame, categories: List[Tuple[str, str]], col_val: str
) -> Tuple[
    List[Tuple[str, str, float, str]], Dict[Union[str, int], List[Union[str, int]]]
]:
    """Function to create directed acyclic graph from given categories specified
    by list of (source,target)
    """
    _levels_df = map(
        lambda x: df.groupby(list(x), as_index=False)[col_val].sum(), categories
    )
    return transform_data_to_source_target(list(_levels_df), categories)


def plot_sankey(
    label: List[str],
    color_node: List[str],
    source: List[Union[str, int]],
    target: List[Union[str, int]],
    values: List[float],
    color_conn: List[str],
    title: str,
) -> None:
    """
    Plot Sankey Diagram
    """
    fig = go.Figure(
        data=[
            go.Sankey(
                valueformat=".0f",
                valuesuffix="Mill R",
                node=dict(
                    pad=5,
                    thickness=10,
                    line=dict(color="black", width=0.05),
                    label=label,
                    # x=[0.1, 0.1, 0.1, 0.1, 0.1, 0.3,0.3,0.3,0.3,0.3,0.3,0.5], # Need to adjust positions
                    # y=[0.1, 0.3, 0.4, 0.5, 0.6, 0.1,0.3,0.4,0.5,0.6,0.7,0.4],
                    color=color_node,
                ),
                link=dict(
                    arrowlen=30,
                    source=source,  # indices correspond to source node wrt to label
                    target=target,
                    value=values,
                    color=color_conn,
                ),
            )
        ]
    )

    fig.update_layout(
        hovermode="x",
        title=title,
        font=dict(size=10, color="black"),
        width=1300,
        height=700,
    )

    fig.show()

    return None


def create_sankey_from_df(
    df: pd.DataFrame, categories: List[Tuple[str, str]], col_val: str, title: str
) -> None:
    """Master function for creating Sankey diagram from dat by aggregation"""
    data_coloured, label_categorized = df_to_dag(df, categories, col_val)

    # # TODO: Involve order services

    # label_categorized["day_name"]=ea.sort_week_day(label_categorized["day_name"])
    # label_categorized["month"]=ea.sort_month(label_categorized["month"])
    # label_categorized["year"]=sorted(label_categorized["year"])

    label = list(chain.from_iterable(label_categorized.values()))

    source = [label.index(x[0]) for x in data_coloured]

    target = [label.index(x[1]) for x in data_coloured]
    values = [x[2] for x in data_coloured]
    color_conn = [x[3] for x in data_coloured]
    color_node = ["#a6cee3"] * len(label)

    plot_sankey(label, color_node, source, target, values, color_conn, title)

    return None

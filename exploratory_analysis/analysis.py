"""Python module for advanced analysis
of a DF
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable, Union, Tuple
from itertools import product
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from exploratory_analysis.preprocessing import one_hot_encode


def show_corr_matrix_filtered(
    df_input: pd.DataFrame,
    li_one_hot: Optional[Union[List[str],Dict[str,str]]] = None,
    threshold_absolute_correlation: Optional[float] = 0.1,
    by_which: Optional[str] = "seaborn",
    label_corr: Optional[bool] = False,
    round_label: Optional[int] = None,
    title_x_axis : Optional[str] = None, title_y_axis: Optional[str] = None
) -> pd.DataFrame:
    """Show correlation matrix filtered by correlations with absolute value > 0.1
    and no feature self correlation

    Args:
        df_input (pd.DataFrame): Input DF
        dummies_dict (Dict[str,List[str]]): Dict with information of dummy columns
            resulting from one-hot-encoding. Form of dictionary:
                column_one_hot_encoded: List of resulting columns
        threshold_absolute_correlation: Threshold absolute value for filtering the correlation

    Returns:
        (pd.DataFrame): Resulting filtered Correlation matrix
    """

    if li_one_hot is not None:
        _one_hot_result = one_hot_encode(df_input=df_input, li_one_hot=li_one_hot)
        _df_input_one_hot = _one_hot_result["df_result"]
        _dummies_dict = _one_hot_result["dummies_dict"]

    _df_corr = _df_input_one_hot.corr(numeric_only=True)

    if round_label is not None:
        _df_corr = _df_corr.round(round_label)

    # Delete pairwise correlation between one-hot-encode columns
    for one_hot_cols in _dummies_dict.values():
        for col_1, col_2 in product(one_hot_cols, one_hot_cols):
            _df_corr.at[col_1, col_2] = np.nan
            _df_corr.at[col_2, col_1] = np.nan

    # Filter correlation matrix
    filtered_df = _df_corr[
        (
            (_df_corr >= threshold_absolute_correlation)
            | (_df_corr <= -threshold_absolute_correlation)
        )
        & (_df_corr != 1)
    ]

    filtered_df = filtered_df.dropna(how="all").dropna(axis=1, how="all")
    
    ## Seaborn 
    if by_which == "seaborn":
        sns.set()
        plt.figure(figsize=(30, 10))
        sns.heatmap(
            filtered_df, annot=label_corr, cmap="Reds", linewidths=0.5, linecolor="gray"
        )
        if title_x_axis is not None:
            plt.xlabel(title_x_axis)
        if title_y_axis is not None:
            plt.ylabel(title_y_axis)
        plt.show()

    ## Plotly    
    elif by_which == "plotly":
        fig = px.imshow(
            filtered_df,
            labels={
                "x": "x-Feature ",
                "y": "y-Feature",
                "color": "Correlation Coefficient",
            },
            text_auto=label_corr,
            width=1000,
            aspect="auto",
        )
        _arg_xaxes=dict(tickangle=45, ticksuffix="  ")
        if title_x_axis is not None:
            _arg_xaxes={**_arg_xaxes, "title": title_x_axis}
        fig.update_xaxes(**_arg_xaxes)

        _arg_yaxes=dict(tickangle=45, ticksuffix="  ")
        if title_x_axis is not None:
            _arg_yaxes={**_arg_yaxes, "title": title_y_axis}
        fig.update_yaxes(**_arg_yaxes)
        fig.update_yaxes(tickangle=-45, ticksuffix="  ")
        fig.update_layout(
            title="Correlation",
            font=dict(
                size=11,
            ),
        )
        fig.show()

    return filtered_df


def show_graph_with_labels(
    adjacency_matrix: np.ndarray,
    mylabels: Dict[int, str],
    centrality: str,
    figsize=(15, 15),
    font_sizes: Optional[Dict[str, int]] = None,
) -> None:
    """Create a graph visualizing the correlation between features.
    Only correlations with absolute value > 0.1 are depicted.
    Red (resp. green) edges are for negative (resp. positive) correlations.
    Width of the edge is proportional to the absolute value of the correlation.
    Size of the node is proportional to the centrality of the node

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix
        mylabels (Dict[int, str]): Labels - Index: Label
        centrality (str): Centrality computation method
        figsize (tuple, optional): Size of the figure.
            Defaults to (15, 15).
        font_sizes (_type_, optional): Font size. Dict
            for node and edge
            Defaults to dict(node=10, edge=8).
    """
    if font_sizes is None:
        font_sizes = {"node": 10, "edge": 8}

    nodes = list(range(len(adjacency_matrix)))
    rows, cols = np.where(adjacency_matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edge_labels = {}
    for edge in edges:
        if adjacency_matrix[edge] < 0:
            color = "r"
        elif adjacency_matrix[edge] > 0:
            color = "g"
        graph.add_edge(
            edge[0], edge[1], color=color, weight=abs(adjacency_matrix[edge] * 10)
        )
        edge_labels[(edge[0], edge[1])] = np.round(adjacency_matrix[edge], 2)
    pos = nx.spring_layout(graph)

    centrality_method = dict(
        degree="degree_centrality",
        load="load_centrality",
        eigenvector="eigenvector_centrality",
    )

    centrality = getattr(nx, centrality_method[centrality])(graph)

    plt.figure(figsize=figsize)
    colors = nx.get_edge_attributes(graph, "color").values()
    weights = nx.get_edge_attributes(graph, "weight").values()

    nx.draw(
        graph,
        pos,
        labels=mylabels,
        node_size=[v * 1000 for v in centrality.values()],
        with_labels=True,
        edge_color=colors,
        width=list(weights),
        font_size=font_sizes["node"],
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=font_sizes["edge"],
        font_color="b",
    )
    plt.show()

    return graph


class GraphFromAdjacencyMatrix:
    """
    Usage:
    def _coloring_by_sign(num_input: float)->str:
    if num_input<= 0:
         return 'r'
    return 'g'

    GraphObject=GraphFromAdjacencyMatrix(
        adjacency_matrix=adjacency_matrix,
        node_labels=dict_labels,centrality='eigenvector',
        func_edge_coloring=_coloring_by_sign,
        func_edge_weight=lambda val: abs(val * 10),
        func_edge_label=lambda val: np.round(val, 2))
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_labels: Dict[int, str],
        func_edge_coloring: Callable[[float], str],
        func_edge_weight: Callable[[float], float],
        func_edge_label: Callable[[float], Union[str, int]],
        adjust_node_size_from_centrality: Optional[bool] = True,
        centrality: Optional[str] = "eigenvector",
    ) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.centrality = centrality

        nodes = list(range(len(adjacency_matrix)))
        rows, cols = np.where(adjacency_matrix != 0)
        edges = zip(rows.tolist(), cols.tolist())

        # Create graph
        self.nx_graph = nx.Graph()
        self.nx_graph.add_nodes_from(nodes)

        for _edge in edges:
            _val_adjacency = adjacency_matrix[_edge]
            self.nx_graph.add_edge(
                *_edge,
                color=func_edge_coloring(_val_adjacency),
                weight=func_edge_weight(_val_adjacency),
                label=func_edge_label(_val_adjacency),
            )

        # Add node label
        nx.set_node_attributes(self.nx_graph, node_labels, name="label")

        # Compute Position
        # Apply algorithm to compute best node placement
        self.pos = nx.spring_layout(self.nx_graph)

        # Add position to graph object
        for _node, _pos in self.pos.items():
            self.nx_graph.nodes[_node]["pos"] = _pos

        # add edge positions to the graph object
        for _edge in self.nx_graph.edges:
            self.nx_graph.edges[_edge]["pos"] = (self.pos[_edge[0]], self.pos[_edge[1]])

        if adjust_node_size_from_centrality is None:
            pass

        centrality_method = dict(
            degree="degree_centrality",
            load="load_centrality",
            eigenvector="eigenvector_centrality",
        )
        nx.set_node_attributes(
            self.nx_graph,
            getattr(nx, centrality_method[centrality])(self.nx_graph),
            name="centrality",
        )

    def plot_sns(
        self,
        figsize: Optional[Tuple[int, int]] = (15, 15),
        font_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        """ Plot method
        """

        if font_sizes is None:
            font_sizes = {"node": 10, "edge": 8}

        plt.figure(figsize=figsize)

        _pos = list(nx.get_node_attributes(self.nx_graph, "pos").values())

        nx.draw(
            self.nx_graph,
            pos=_pos,
            labels=nx.get_node_attributes(self.nx_graph, "label"),
            node_size=list(nx.get_node_attributes(self.nx_graph, "size").values()),
            with_labels=True,
            edge_color=list(nx.get_edge_attributes(self.nx_graph, "color").values()),
            width=list(nx.get_edge_attributes(self.nx_graph, "weight").values()),
            font_size=font_sizes["node"],
        )
        nx.draw_networkx_edge_labels(
            self.nx_graph,
            pos=_pos,
            edge_labels=nx.get_edge_attributes(self.nx_graph, "label"),
            font_size=font_sizes["edge"],
            font_color="b",
        )
        plt.show()

        return None

    def _create_plotly_nodes(self) -> None:

        node_trace = go.Scatter(
            x=[
                _pos[0]
                for _pos in nx.get_node_attributes(self.nx_graph, "pos").values()
            ],
            y=[
                _pos[1]
                for _pos in nx.get_node_attributes(self.nx_graph, "pos").values()
            ],
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15, title="Centrality", xanchor="left", titleside="right"
                ),
                line_width=2,
            ),
        )

        node_text = []

        node_adjacencies = []
        for adjacencies in self.nx_graph.adjacency():
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append("# of connections: " + str(len(adjacencies[1])))

        print(nx.get_node_attributes(self.nx_graph, "size"))

        node_trace.marker.color = list(
            nx.get_node_attributes(self.nx_graph, "centrality").values()
        )
        node_trace.marker.size = [
            centrality * 100
            for centrality in nx.get_node_attributes(
                self.nx_graph, "centrality"
            ).values()
        ]
        node_trace.text = [
            f"Graph Centrality: {centrality:.2f}"
            for centrality in nx.get_node_attributes(
                self.nx_graph, "centrality"
            ).values()
        ]

        return node_trace

    def _create_plotly_edges(self) -> None:

        edge_x = []
        edge_y = []

        edge_text = []
        for edge in self.nx_graph.edges():
            _x0, _y0 = self.nx_graph.nodes[edge[0]]["pos"]
            _x1, _y1 = self.nx_graph.nodes[edge[1]]["pos"]
            edge_x.extend([_x0, _x1, None])
            edge_y.extend([_y0, _y1, None])
            edge_text.append(
                f'Correlation Coefficient: {nx.get_edge_attributes(self.nx_graph,"label")[edge]}'
            )

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="text",
            mode="lines",
        )

        # edge_trace.text=edge_text
        # edge_trace.marker.size = list(nx.get_edge_attributes(self.nx_graph,"weight").values())

        return edge_trace

    def plot_plotly(self) -> None:
        """ Plot by plotly
        """
        fig = go.Figure(
            data=[self._create_plotly_nodes(), self._create_plotly_edges()],
            layout=go.Layout(
                title="<br>Hier ein Titel",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Hier kann ein Text stehen",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        fig.show()


class CorrelationFeatures:
    """Class for analyzing correlation between the features
    in a DF
    """

    def __init__(
        self,
        df_input: pd.DataFrame,
        dummies_dict: Dict[str, List[str]],
        filtered_df: Optional[pd.DataFrame] = None,
        filter_option: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialization

        Args:
            df_input (pd.DataFrame): DF to be analyzed
            dummies_dict (Dict[str, List[str]]): dummies_dict (Dict[str,List[str]]):
                Dict with information of dummy columns
                resulting from one-hot-encoding. Form of dictionary:
                    key: column_one_hot_encoded
                    value: List of resulting columns
        """

        self.df_input = df_input
        self.dummies_dict = dummies_dict

        # Compute the correlation between the features
        # The correlation between one-hot-encoded features will not be shown (a.k.a. set = nan)
        self.df_corr = df_input.corr(numeric_only=True)
        for one_hot_cols in dummies_dict.values():
            for col_1, col_2 in product(one_hot_cols, one_hot_cols):
                self.df_corr.at[col_1, col_2] = np.nan
                self.df_corr.at[col_2, col_1] = np.nan

        # Variable for the filtered correlation matrix in DF
        self.filtered_df = filtered_df

        # Variable for the choosen filter
        self.filter_option = {} if filter_option is None else filter_option

    def filter_correlations(
        self, threshold_absolute_correlation: float
    ) -> CorrelationFeatures:
        """Filter the correlation matrix given certain
        threshold value.
        Args:
            threshold_absolute_correlation (float): Entries whose absolute
                value > this value will be filtered out
        Returns:
            CorrelationFeatures: Class with filtered correlation matrix
        """

        assert (
            threshold_absolute_correlation > 0
        ), "threshold_absolute correlation needs to be positive!"

        self.filtered_df = self.df_corr[
            (
                (self.df_corr >= threshold_absolute_correlation)
                | (self.df_corr <= -threshold_absolute_correlation)
            )
            & (self.df_corr != 1.000)
        ]

        self.filtered_df = self.filtered_df.dropna(how="all").dropna(axis=1, how="all")

        self.filter_option[
            "threshold_absolute_correlation"
        ] = threshold_absolute_correlation

        return CorrelationFeatures(
            df_input=self.df_input,
            dummies_dict=self.dummies_dict,
            filtered_df=self.filtered_df,
            filter_option=self.filter_option,
        )

    def show_heat_map(self, filtered: Optional[bool] = True) -> None:
        """Function for plotting the correlation matrix as a heat map

        Args:
            filtered (Optional[bool], optional): Whether filtered or
                unfiltered matrix should be plotted. Defaults to True.
        """

        corr_to_be_plotted = self.filtered_df if filtered else self.df_corr

        sns.set()
        plt.figure(figsize=(30, 10))
        sns.heatmap(
            corr_to_be_plotted,
            annot=True,
            cmap="Reds",
            linewidths=0.5,
            linecolor="gray",
        )
        plt.show()

    def show_correlation_graph(self, filtered: Optional[bool] = True):
        """Show the correlation graph

        Args:
            filtered (Optional[bool], optional): Whether to consider filtered correlation matrix.
                Defaults to True.
        """

        adjacency_matrix = self.filtered_df.fillna(0) if filtered else self.df_corr

        dict_labels = {
            list(adjacency_matrix.columns).index(col): col
            for col in list(adjacency_matrix.columns)
        }

        adjacency_matrix = adjacency_matrix.to_numpy()

        show_graph_with_labels(adjacency_matrix, dict_labels, "eigenvector")


# Feature selection


class FeatureSelector:
    """Class for feature selection"""

    def __init__(
        self,
        df: pd.DataFrame,  # pylint: disable=invalid-name
        target: str,
        num_feature_keep: Optional[int] = None,  # pylint: disable=invalid-name
    ) -> None:
        """Initialization

        Args:
            df (pd.DataFrame): DF to be analyzed
            target (str): Target column
            num_feature_keep (Optional[int], optional): Number of features to keep after selection.
                Defaults to None.
        """

        self.df = df  # pylint: disable=invalid-name
        self.target = target

        self.num_feature_keep = num_feature_keep

        # Create feature and target DF
        self.df_target = self.df[target]
        self.df_features = self.df.drop(columns=[target])

        # Initiate the feature selector
        self.selector = SelectKBest(
            f_classif,
            k=self.num_feature_keep if self.num_feature_keep is not None else "all",
        )  # k is the number of features to be selected

        # Train the selector
        self.selector.fit_transform(self.df_features, self.df_target)

        # Get the index of the columns kept after applying the feature selector
        self.support = self.selector.get_support()

        # Extract the corresponding feature names
        self.features_names_selected = self.df_features.loc[
            :, self.support
        ].columns.tolist()

        # Create new DF by selected index
        self.df_selected = self.df[[*self.features_names_selected, target]]

        # Scores & P-Values
        self.scores = self.selector.scores_
        self.p_values = self.selector.pvalues_

    @property
    def result_table(self) -> pd.DataFrame:
        """DF Containing scores and p_values of the columns"""
        return pd.DataFrame(
            zip(
                self.df_features.columns, self.selector.scores_, self.selector.pvalues_
            ),
            columns=["column", "scores", "p_values"],
        )

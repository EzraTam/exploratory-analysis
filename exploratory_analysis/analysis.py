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
    li_one_hot: Optional[Union[List[str], Dict[str, str]]] = None,
    threshold_absolute_correlation: Optional[float] = 0.1,
    by_which: Optional[str] = "seaborn",
    label_corr: Optional[bool] = False,
    round_label: Optional[int] = None,
    title_x_axis: Optional[str] = None,
    title_y_axis: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
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
    _df_input_one_hot = df_input
    one_hot_result = None
    if li_one_hot is not None:
        one_hot_result = one_hot_encode(
            df_input=_df_input_one_hot, li_one_hot=li_one_hot
        )
        _df_input_one_hot = one_hot_result["df_result"]
        _dummies_dict = one_hot_result["dummies_dict"]

    _df_corr = _df_input_one_hot.corr(numeric_only=True)

    if round_label is not None:
        _df_corr = _df_corr.round(round_label)

    if one_hot_result is not None:
        # Delete pairwise correlation between one-hot features
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
        _arg_xaxes = dict(tickangle=45, ticksuffix="  ")
        if title_x_axis is not None:
            _arg_xaxes = {**_arg_xaxes, "title": title_x_axis}
        fig.update_xaxes(**_arg_xaxes)

        _arg_yaxes = dict(tickangle=45, ticksuffix="  ")
        if title_x_axis is not None:
            _arg_yaxes = {**_arg_yaxes, "title": title_y_axis}
        fig.update_yaxes(**_arg_yaxes)
        fig.update_yaxes(tickangle=-45, ticksuffix="  ")
        fig.update_layout(
            title="Correlation",
            font=dict(
                size=11,
            ),
        )
        fig.show()

    return dict(corr_filtered=filtered_df, one_hot_result=one_hot_result)


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
        df_adjacency: Optional[pd.DataFrame] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        node_labels: Optional[Dict[int, str]] = None,
        func_edge_coloring: Optional[Callable[[float], str]] = lambda val: "red"
        if val <= 0
        else "green",
        func_edge_weight: Optional[Callable[[float], float]] = lambda val: abs(val * 2),
        adjust_node_size_from_centrality: Optional[bool] = True,
        centrality: Optional[str] = "eigenvector",
        optimal_distance_nodes: Optional[float] = None,
    ) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.centrality = centrality
        if df_adjacency is not None:
            node_labels = {
                list(df_adjacency.columns).index(col): col
                for col in list(df_adjacency.columns)
            }
            self.adjacency_matrix = df_adjacency.to_numpy()

        nodes = list(range(len(self.adjacency_matrix)))
        rows, cols = np.where(self.adjacency_matrix != 0)
        edges = zip(rows.tolist(), cols.tolist())

        # Create graph
        self.nx_graph = nx.Graph()
        self.nx_graph.add_nodes_from(nodes)

        for _edge in edges:
            _val_adjacency = self.adjacency_matrix[_edge]
            self.nx_graph.add_edge(
                *_edge,
                color=func_edge_coloring(_val_adjacency),
                weight=func_edge_weight(_val_adjacency),
                label=f"Correlation between: {node_labels[_edge[0]]} and {node_labels[_edge[1]]}<br>Correlation Coefficient: {_val_adjacency}",
            )

        # Add node label
        nx.set_node_attributes(self.nx_graph, node_labels, name="label")

        # Compute Position
        # Apply algorithm to compute best node placement
        self.pos = nx.spring_layout(self.nx_graph, k=optimal_distance_nodes)

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
        """Plot method"""

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
        _label_names = nx.get_node_attributes(self.nx_graph, "label")
        for adjacencies in self.nx_graph.adjacency():
            node_adjacencies.append(len(adjacencies[1]))
            _node_text = f"Node Name: {_label_names[adjacencies[0]]}<br>"
            _node_text += f"# of connections: {len(adjacencies[1])}<br>"
            node_text.append(_node_text)

        node_text = [
            _node_text + f"Centrality: {centrality:.2f}"
            for centrality, _node_text in zip(
                nx.get_node_attributes(self.nx_graph, "centrality").values(), node_text
            )
        ]

        node_trace.marker.color = list(
            nx.get_node_attributes(self.nx_graph, "centrality").values()
        )
        node_trace.marker.size = [
            centrality * 100
            for centrality in nx.get_node_attributes(
                self.nx_graph, "centrality"
            ).values()
        ]
        node_trace.text = node_text

        return node_trace

    def _create_plotly_edges(self) -> None:

        middle_node_traces = []
        edge_traces = []

        for _edge in self.nx_graph.edges():
            _x0, _y0 = self.nx_graph.nodes[_edge[0]]["pos"]
            _x1, _y1 = self.nx_graph.nodes[_edge[1]]["pos"]

            edge_traces.append(go.Scatter(
                x=[_x0, _x1, None],
                y=[_y0, _y1, None],
                line=dict(
                    width=nx.get_edge_attributes(self.nx_graph, "weight")[_edge],
                    color=nx.get_edge_attributes(self.nx_graph, "color")[_edge]),
                mode="lines",
            ))

            # Label for edges
            middle_node_traces.append(
                go.Scatter(
                    x=[(_x0 + _x1) / 2],
                    y=[(_y0 + _y1) / 2],
                    text=nx.get_edge_attributes(self.nx_graph,"label")[_edge],
                    mode="markers",
                    hoverinfo="text",
                    marker=go.scatter.Marker(opacity=0),
                    hoverlabel=dict(
                        bgcolor=nx.get_edge_attributes(self.nx_graph, "color")[_edge]),
                )
            )

        return {"edges": edge_traces, "middle_nodes": middle_node_traces}

    def plot_plotly(self, plot_title: Optional[str] = "", plot_description:Optional[str]="") -> None:
        """Plot by plotly"""
        fig = go.Figure(
            layout=go.Layout(
                title=plot_title,
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=plot_description,
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

        for _edge in self._create_plotly_edges()["edges"]:
            fig.add_trace(_edge)

        for _node in self._create_plotly_edges()["middle_nodes"]:
            fig.add_trace(_node)
        
        fig.add_trace(self._create_plotly_nodes())

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

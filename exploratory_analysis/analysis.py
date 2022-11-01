"""Python module for advanced analysis
of a DF
"""

from typing import Dict, List
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def show_corr_matrix_filtered(
    df_input: pd.DataFrame, dummies_dict: Dict[str, List[str]]
) -> pd.DataFrame:
    """Show correlation matrix filtered by correlations with absolute value > 0.1
    and no feature self correlation

    Args:
        df_input (pd.DataFrame): Input DF
        dummies_dict (Dict[str,List[str]]): Dict with information of dummy columns
            resulting from one-hot-encoding. Form of dictionary:
                column_one_hot_encoded: List of resulting columns

    Returns:
        (pd.DataFrame): Resulting filtered Correlation matrix
    """
    df_corr = df_input.corr(numeric_only=True)
    for one_hot_cols in dummies_dict.values():
        for col_1 in one_hot_cols:
            for col_2 in one_hot_cols:
                df_corr.at[col_1, col_2] = np.nan
                df_corr.at[col_2, col_1] = np.nan
    filtered_df = df_corr[((df_corr >= 0.1) | (df_corr <= -0.1)) & (df_corr != 1.000)]
    filtered_df = filtered_df.dropna(how="all").dropna(axis=1, how="all")
    sns.set()
    plt.figure(figsize=(30, 10))
    sns.heatmap(filtered_df, annot=True, cmap="Reds", linewidths=0.5, linecolor="gray")
    plt.show()
    return filtered_df


def show_graph_with_labels(
    adjacency_matrix: np.ndarray,
    mylabels: Dict[int, str],
    centrality: str,
    figsize=(15, 15),
    font_sizes=dict(node=10, edge=8),
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

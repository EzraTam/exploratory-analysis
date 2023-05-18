from typing import Iterable, Union, Optional, Tuple, Dict, Any
import networkx as nx

NodeType = Union[int,str]
EdgeType = Tuple[NodeType,NodeType]

class Graph:

    def __init__(
        self,
        nodes : Iterable[NodeType],
        edges : Iterable[EdgeType],
        edge_properties: Optional[Dict[EdgeType,Any]] = None, # Consider to specify the Any-Types
        weights : Optional[Dict[EdgeType,EdgeType]]=None,
        colors : Optional[Iterable[str]]=None,
        compute_pos: Optional[bool]=True
        ) -> None:
        """
        edge_properties: dictionary with keys == edge and values equal to a
        dictionary containing ...
        """

        self.nodes = nodes
        
        self.nx_object : nx.Graph = nx.Graph()

        # Add nodes
        self.nx_object.add_nodes_from(nodes)

        if edge_properties is not None:
            =map(edge_properties.values())
        
        # Add Edges
        for edge, color, weight in zip(edges,colors,weights):
            self.nx_object.add_edge(
                edge[0],
                edge[1],
                color=color, 
                weight=weight
            )
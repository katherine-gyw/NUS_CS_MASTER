import networkx as nx
import numpy as np
from networkx.algorithms import tree
from networkx.algorithms.clique import find_cliques

from factor import Factor
from factor_utils import factor_product

""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """


""" END ADD HELPER FUNCTIONS HERE """


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]

    """ YOUR CODE HERE """
    pos_ls = [False] * len(jt_cliques)
    for factor in factors:
        for i, clique in enumerate(jt_cliques):
            if set(factor.var).issubset(clique):
                clique_factors[i] = factor_product(clique_factors[i], factor)
                break

    """ END YOUR CODE HERE """
    assert len(clique_factors) == len(
        jt_cliques
    ), "there should be equal number of cliques and clique factors"
    return clique_factors


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) in the junction tree and
    the list of edges between cliques in the junction tree. [i, j] in jt_edges means that cliques[j] is a neighbor
    of cliques[i] and vice versa. [j, i] should also be included in the numpy array of edges if [i, j] is present.
    You can use nx.Graph() and nx.find_cliques().

    Args:
        nodes: numpy array of nodes [x1, ..., xN]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors.

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2], ...]
        numpy array of junction tree edges e.g. [[0,1], ...], [i,j] means that cliques[i]
            and cliques[j] are neighbors.
    """
    jt_cliques = []
    jt_edges = np.array(edges)  # dummy value

    """ YOUR CODE HERE """
    G1 = nx.empty_graph(nodes)
    for edge in edges:
        G1.add_edge(edge[0], edge[1])
    jt_cliques = list(find_cliques(G1))
    edge_ls = []
    for i, clique1 in enumerate(jt_cliques):
        for j, clique2 in enumerate(jt_cliques[i:]):
            if len(set(clique1).intersection(set(clique2))) > 0 and j != 0:
                edge_ls.append([i, i + j])
    G = nx.empty_graph(len(jt_cliques))
    for edge in edge_ls:
        weight = len(set(jt_cliques[edge[0]]).intersection(jt_cliques[edge[1]]))
        G.add_edge(edge[0], edge[1], weight=weight)
    T = tree.maximum_spanning_edges(G, algorithm="kruskal", data=False, ignore_nan=True)
    jt_edges = np.array(list(T))
    """ END YOUR CODE HERE """
    return jt_cliques, jt_edges


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns its the cliques, edges and clique factors in the junction tree.
    DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors

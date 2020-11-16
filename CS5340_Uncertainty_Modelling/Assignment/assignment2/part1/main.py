""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X
"""

import json
import os
from argparse import ArgumentParser

import networkx as nx
import numpy as np

from factor import Factor
from factor_utils import factor_evidence, factor_marginalize, factor_product
from jt_construction import construct_junction_tree

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")  # we will store the input data files here!
PREDICTION_DIR = os.path.join(
    DATA_DIR, "predictions"
)  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """


def get_neighrbor(edge_arr):
    edge_dict = {}
    for edge in edge_arr:
        if edge[0] not in edge_dict.keys():
            edge_dict[edge[0]] = [edge[1]]
        else:
            edge_dict[edge[0]].append(edge[1])
        if edge[1] not in edge_dict.keys():
            edge_dict[edge[1]] = [edge[0]]
        else:
            edge_dict[edge[1]].append(edge[0])
    return edge_dict


def find_edges(nodes, all_edges):
    return np.asarray(
        [edge for edge in all_edges if edge[0] in nodes or edge[1] in nodes]
    )


def find_factors(nodes, all_factors):
    return [
        factor
        for factor in all_factors
        if len(set(factor.var).intersection(set(nodes))) > 0
    ]


""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = factors

    """ YOUR CODE HERE """
    factor_ls = []
    node_ls = []
    for i, factor in enumerate(updated_factors):
        cur_factor = factor_evidence(factor, evidence)
        if not cur_factor.is_empty():
            factor_ls.append(cur_factor)
            node_ls = node_ls + list(cur_factor.var)
    updated_factors = factor_ls
    query_nodes = np.array(sorted(np.unique(node_ls)))
    evidence_ls = list(evidence.keys())
    new_edge_ls = []
    for edge in list(updated_edges):
        if edge[0] not in evidence_ls and edge[1] not in evidence_ls:
            new_edge_ls.append(edge)
    updated_edges = np.array(new_edge_ls)
    """ END YOUR CODE HERE """
    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    num_nodes = len(jt_clique_factors)
    if num_nodes == 1:
        return clique_potentials
    messages = [[None] * num_nodes for _ in range(num_nodes)]
    neighbors_dict = get_neighrbor(jt_edges)
    root = jt_edges[0][0]

    def collect(i, j):
        cur_neighbors = neighbors_dict[j]
        for k in cur_neighbors:
            if k != i:
                collect(j, k)
        send_message(j, i)

    def distribute(i, j):
        cur_neighbor = neighbors_dict[j]
        send_message(i, j)
        for k in cur_neighbor:
            if k != i:
                distribute(j, k)

    def send_message(j, i):
        m = Factor()
        neighbor_ls = neighbors_dict[j]
        for neighbor in neighbor_ls:
            if neighbor != i:
                m = factor_product(m, messages[neighbor][j])
        m = factor_product(m, jt_clique_factors[j])
        Sij = set(jt_clique_factors[i].var).intersection(set(jt_clique_factors[j].var))
        Cj = set(jt_clique_factors[j].var)
        marginalize_ls = list(Cj.difference(Sij))
        if len(marginalize_ls) > 0:
            messages[j][i] = factor_marginalize(m, np.array(marginalize_ls))
        else:
            messages[j][i] = m

    def compute_marginal(i):
        m = Factor()
        neighbor_ls = neighbors_dict[i]
        for neighbor in neighbor_ls:
            m = factor_product(m, messages[neighbor][i])
        m = factor_product(m, jt_clique_factors[i])
        return m

    for node in neighbors_dict[root]:
        collect(root, node)

    for node in neighbors_dict[root]:
        distribute(root, node)

    for i in range(len(jt_clique_factors)):
        clique_potentials[i] = compute_marginal(i)

    """ END YOUR CODE HERE """
    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = []

    """ YOUR CODE HERE """
    for node in list(query_nodes):
        m = Factor()
        for potential in clique_potentials:
            if node in potential.var and len(potential.var) > 1:
                marginalize_ls = list(potential.var)
                marginalize_ls.remove(node)
                m = factor_marginalize(potential, marginalize_ls)
                break
        m.val = m.val / sum(m.val)
        query_marginal_probabilities.append(m)
    """ END YOUR CODE HERE """
    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_factors = _update_mrf_w_evidence(
        all_nodes=all_nodes, evidence=evidence, edges=edges, factors=factors
    )

    G = nx.empty_graph(query_nodes)
    for edge in updated_edges:
        G.add_edge(edge[0], edge[1])

    # Accounting for evidence could cause graph to be disconnected
    # list of list of nodes
    graph_groups = sorted(nx.connected_components(G), key=len, reverse=True)

    updated_node_marginals = []
    query_nodes = []
    for node_ls in graph_groups:

        node_ls = list(node_ls)
        cur_nodes = np.asarray(node_ls)
        cur_edges = find_edges(node_ls, updated_edges)
        cur_factors = find_factors(node_ls, updated_factors)

        if len(node_ls) == 1:

            assert len(cur_factors) == 1
            assert len(cur_edges) == 0

            cur_factors[0].val = cur_factors[0].val / sum(cur_factors[0].val)
            cur_query_node_marginals = cur_factors

        else:

            jt_cliques, jt_edges, jt_factors = construct_junction_tree(
                nodes=cur_nodes, edges=cur_edges, factors=cur_factors
            )

            clique_potentials = _get_clique_potentials(
                jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors
            )

            cur_query_node_marginals = _get_node_marginal_probabilities(
                query_nodes=cur_nodes,
                cliques=jt_cliques,
                clique_potentials=clique_potentials,
            )

        updated_node_marginals = updated_node_marginals + cur_query_node_marginals
        cur_query_node = []
        for item in cur_query_node_marginals:
            cur_query_node.append(item.var)

        query_nodes = query_nodes + cur_query_node

    return query_nodes, updated_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, "r") as f:
        input_config = json.load(f)

    nodes = np.array(input_config["nodes"])
    edges = np.array(input_config["edges"])

    # parse evidence
    raw_evidence = input_config["evidence"]
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config["factors"]
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(
            var=np.array(raw_factor["var"]),
            card=np.array(raw_factor["card"]),
            val=np.array(raw_factor["val"]),
        )
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    # argparser = ArgumentParser()
    # argparser.add_argument('--case', type=int, required=True,
    #                        help='case number to create observations e.g. 1 if 1.json')
    # args = argparser.parse_args()

    # case = args.case
    case = "4"
    input_file = os.path.join(INPUT_DIR, "{}.json".format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(
        all_nodes=nodes, edges=edges, factors=factors, evidence=evidence
    )

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, "{}.json".format(case))
    with open(prediction_file, "w") as f:
        json.dump(predictions, f, indent=1)
    print(
        "INFO: Results for test case {} are stored in {}".format(case, prediction_file)
    )


if __name__ == "__main__":
    main()

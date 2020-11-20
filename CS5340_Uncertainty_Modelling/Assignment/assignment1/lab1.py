""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: GUO YUEWEN
Email: e0575795@u.nus.edu
"""

import copy
from typing import List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card
    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    out.val = A.val[idxA] * B.val[idxB]
    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """
    out.var = np.array([x for x in factor.var if x not in var])
    out.card = np.array([c for v, c in zip(factor.var, factor.card) if v not in var])
    out.val = np.zeros(np.prod(out.card))

    var_assignments = factor.get_all_assignments()
    col_idx = []
    for i in range(len(factor.var)):
        if factor.var[i] not in var:
            col_idx.append(i)
    for i in range(var_assignments.shape[0]):
        out_index = assignment_to_index(var_assignments[i][col_idx], out.card)
        out.val[out_index] += factor.val[i]
    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the
    evidence to zero.
    """
    evidence_feature_ls = list(evidence.keys())
    for i, factor in enumerate(out):
        factor_ls = list(factor.var)
        feature_idx = []
        evidence_val_ls = []
        for idx, feature in enumerate(evidence_feature_ls):
            if feature in factor_ls:
                evidence_val_ls.append(evidence[feature])
                feature_idx.append(factor_ls.index(feature))
        if len(feature_idx)==0:
            continue
        factor_indices = factor.get_all_assignments()
        for j,index in enumerate(factor_indices):
            if sum(index[feature_idx]!=evidence_val_ls)>0:
                out[i].val[j] = 0
    return out


"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = A.val[idxA] + B.val[idxB]

    if A.val_argmax is not None or B.val_argmax is not None:
        out.val_argmax = [{} for _ in range(len(out.val))]
        if A.val_argmax is not None:
            for i, j in enumerate(idxA):
                out.val_argmax[i].update(A.val_argmax[j])
        if B.val_argmax is not None:
            for i, j in enumerate(idxB):
                out.val_argmax[i].update(B.val_argmax[j])

    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var.
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """
    out.var = np.array([x for x in factor.var if x not in var])
    out.card = np.array([c for v, c in zip(factor.var, factor.card) if v not in var])
    out.val = np.zeros(np.prod(out.card))-100000000000

    var_assignments = factor.get_all_assignments()
    col_idx = []  # Col that are not marginalized out
    var_idx = {}  # Mapper to columns to be marginalized
    for i in range(len(factor.var)):
        if factor.var[i] not in var:
            col_idx.append(i)
        else:
            var_idx[factor.var[i]] = i
    val_argmax_ls = []
    for i in range(len(out.val)):
        val_argmax_ls.append({})

    for i in range(var_assignments.shape[0]):
        out_index = assignment_to_index(var_assignments[i][col_idx], out.card)
        if factor.val[i] > out.val[out_index]:
            out.val[out_index] = factor.val[i]
            val_argmax_ls[out_index] = {
                item: var_assignments[i][var_idx[item]] for item in var
            }
            if factor.val_argmax is not None:
                val_argmax_ls[out_index].update(factor.val_argmax[i])
    out.val_argmax = val_argmax_ls
    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    for factor in factors:
        joint = factor_product(joint, factor)
    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """

    for factor in factors:
        output = factor_product(output, factor)
    output = observe_evidence([output], evidence)
    var_ls = list(output[0].var)
    var_ls.remove(V)
    output = factor_marginalize(output[0], var_ls)
    output.val = output.val/sum(output.val)
    return output


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Note that we create an undirected graph regardless of the input graph since
    # 1) this facilitates graph traversal
    # 2) the algorithm for undirected and directed graphs is essentially the same for tree-like graphs.
    visualize_graph(graph)

    # You can use any node as the root since the graph is a tree.
    # For simplicity we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.

    Hint: You might find it useful to add auxilliary functions. You may add
      them as either inner (nested) or external functions.
    """

    def collect(i, j):
        cur_neighbors = graph.neighbors(j)
        for k in cur_neighbors:
            if k != i:
                collect(j, k)
        send_message(j, i)

    def distribute(i, j):
        cur_neighbor = graph.neighbors(j)
        send_message(i, j)
        for k in cur_neighbor:
            if k != i:
                distribute(j, k)

    def send_message(j, i):
        m = Factor()
        neighbor_ls = graph.neighbors(j)
        for neighbor in neighbor_ls:
            if neighbor != i:
                m = factor_product(m, messages[neighbor][j])
        if 'factor' in graph.nodes[j].keys():
            m = factor_product(m, graph.nodes[j]['factor'])
        if 'factor' in graph.edges[i,j].keys():
            m = factor_product(m, graph.edges[i,j]['factor'])
        messages[j][i] = factor_marginalize(m, [j])

    def compute_marginal(i):
        m = Factor()
        neighbor_ls = graph.neighbors(i)
        for neighbor in neighbor_ls:
            m = factor_product(m, messages[neighbor][i])
        if 'factor' in graph.nodes[i]:
            m = factor_product(m, graph.nodes[i]['factor'])
        m.val = m.val/sum(m.val)
        return m

    for node in graph.neighbors(root):
        collect(root, node)

    for node in graph.neighbors(root):
        distribute(root, node)

    for i in V:
        marginals.append(compute_marginal(i))
    return marginals


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """
    max_decoding = {}
    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """
    factors = observe_evidence(factors, evidence)
    for factor in factors:
        factor.val = np.log(factor.val)
    graph = generate_graph_from_factors(factors)
    root = 0
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    def collect(i, j):
        for k in graph.neighbors(j):
            if k != i:
                collect(j, k)
        send_message(j, i)

    def distribute(i, j):
        for k in graph.neighbors(j):
            if k != i:
                distribute(j, k)

    def send_message(j, i):
        m = Factor()
        for neighbor in graph.neighbors(j):
            if neighbor != i:
                m = factor_sum(m, messages[neighbor][j])
        if 'factor' in graph.nodes[j].keys():
            m = factor_sum(m, graph.nodes[j]['factor'])
        if 'factor' in graph.edges[i,j].keys():
            m = factor_sum(m, graph.edges[i,j]['factor'])
        messages[j][i] = factor_max_marginalize(m, [j])

    for node in graph.neighbors(root):
        collect(root, node)

    prod_factor = Factor()
    for neighbor in graph.neighbors(root):
        prod_factor = factor_sum(prod_factor, messages[neighbor][root])
    if "factor" in graph.nodes[root].keys():
        prod_factor = factor_sum(prod_factor, graph.nodes[root]["factor"])
    log_prob_max = np.max(prod_factor.val)
    max_decoding.update({root: np.argmax(prod_factor.val)})
    max_decoding.update(prod_factor.val_argmax[max_decoding[root]])
    for e in evidence:
        max_decoding.pop(e)

    for node in graph.neighbors(root):
        distribute(root, node)
    return max_decoding, log_prob_max

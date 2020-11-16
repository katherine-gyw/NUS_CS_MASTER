# taken from part 1
import copy

import numpy as np

from factor import Factor, assignment_to_index, index_to_assignment


def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)

    Args:
        A: first Factor
        B: second Factor

    Returns:
        Returns the factor product of A and B
    """
    out = Factor()

    """ YOUR CODE HERE """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    out.var = np.union1d(A.var, B.var)

    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    out.val = A.val[idxA] * B.val[idxB]
    """ END YOUR CODE HERE """
    return out


def factor_marginalize(factor, var):
    """
    Returns factor after variables in var have been marginalized out.

    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over

    Returns:
        marginalized factor
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE
     HINT: Use the code from lab1 """
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
    """ END YOUR CODE HERE """
    return out


def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    if evidence is None:
        return out

    observed_vars = list(evidence.keys())
    observed_assignments = [evidence[obs] for obs in observed_vars]

    observed_var_indices = []
    for obs in observed_vars:
        found = np.where(out.var == obs)[0]
        if len(found):
            observed_var_indices.append(found[0])

    if len(observed_var_indices) == 0:
        return out

    all_assignments = out.get_all_assignments()

    # Extract rows where the observed variables are equal to the observed assignments
    out.val = out.val[
        np.all(all_assignments[:, observed_var_indices] == observed_assignments, axis=1)
    ]

    non_observed_indices = [
        i for i in range(len(out.card)) if i not in observed_var_indices
    ]
    out.var = out.var[non_observed_indices]
    out.card = out.card[non_observed_indices]
    """ END YOUR CODE HERE """
    return out

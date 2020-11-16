""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X
"""

import os
import json
import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_product, factor_marginalize
from factor import Factor, assignment_to_index, index_to_assignment
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """
def cal_prob(sample, factors):
    prob = 1
    for key,factor in factors.items():
        sample_ls = [sample[item] for item in list(factor.var)]
        cur_idx = assignment_to_index(np.array(sample_ls), factor.card)
        prob *= factor.val[cur_idx]
    return prob


def cal_q_prob(sample, factors):
    prob = 1
    unused_var = list(sample.keys())

    for key, factor in factors.items():
        if len(set(factor.var).intersection(set(unused_var))) == 0:
            continue
        else:
            for item in factor.var:
                if item in unused_var:
                    unused_var.remove(item)
        sample_ls = [sample[item] for item in list(factor.var)]
        cur_idx = assignment_to_index(np.array(sample_ls), factor.card)
        prob *= factor.val[cur_idx]
    return prob


""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """
    factors = proposal_factors.copy()
    for item in nodes:
        cur_factor = Factor()
        key_ls = []
        for key, factor in factors.items():
            if item in factor.var:
                cur_factor = factor_product(cur_factor, factor)
                key_ls.append(key)
        var_ls = list(cur_factor.var)
        if len(var_ls)>1:
            var_ls.remove(item)
            cur_factor = factor_marginalize(cur_factor, np.array(var_ls))
        cur_sample = np.random.choice(np.arange(cur_factor.card[0]), p=cur_factor.val/sum(cur_factor.val))
        samples[item] = cur_sample
        for key in key_ls:
            factors[key] = factor_evidence(factors[key], {item: cur_sample})
    """ END YOUR CODE HERE """
    assert len(samples.keys()) == len(nodes)
    return samples


def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    topological_order = []
    card_order = {}
    for key, factor in proposal_factors.items():
        topological_order += list(factor.var)
        for i, item in enumerate(list(factor.var)):
            card_order[item] = factor.card[i]
        if len(set(factor.var).intersection(set(evidence.keys()))) > 0:
            cur_factor = factor_evidence(factor, evidence)
            proposal_factors[key] = cur_factor
    topological_order = sorted(np.unique(topological_order))
    for item in evidence.keys():
        topological_order.remove(item)
    card_ls = []
    for item in topological_order:
        card_ls.append(card_order[item])
    out.var = np.array(topological_order)
    out.card = np.array(card_ls)

    px_ls, qx_ls, sample_ls = [], [], []
    # generate sample
    for i in tqdm(range(num_iterations)):
        sample = _sample_step(topological_order, proposal_factors)
        # calculate p(x) from target distribution
        px = cal_prob(dict(list(sample.items()) + list(evidence.items())), target_factors)
        px_ls.append(px)
        # calculate q(x) from proposal distribution
        qx = cal_q_prob(sample, proposal_factors)
        qx_ls.append(qx)
        # record the sample result
        sample_ls.append(list(sample.values()))
    r = np.array(px_ls)/np.array(qx_ls)
    w = r/sum(r)

    # define out values
    card_num = 1
    for item in out.card:
        card_num *= item
    val = np.zeros(card_num)
    for i, item in enumerate(sample_ls):
        idx = assignment_to_index(np.array(item), out.card)
        val[idx] += w[i]
    val /= sum(val)
    out.val = val
    """ END YOUR CODE HERE """
    return out


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()

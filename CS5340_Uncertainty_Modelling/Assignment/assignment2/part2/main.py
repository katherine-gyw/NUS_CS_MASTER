""" CS5340 Lab 2 Part 2: Parameter Learning
See accompanying PDF for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X
"""

import os
import numpy as np
import json

import networkx as nx
from argparse import ArgumentParser

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')  # we will store the input data files here!
OBSERVATION_DIR = os.path.join(DATA_DIR, 'observations')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """

""" D ADD HELPER FUNCTIONS HERE """


def _learn_node_parameter_w(outputs, inputs=None):
    """
    Returns the weight parameters of the linear Gaussian [w0, w1, ..., wI], where I is the number of inputs. Students
    are encouraged to use numpy.linalg.solve() to get the weights. Learns weights for one node only.
    Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        inputs: N x I numpy array of input observations to the linear Gaussian model

    Returns:
        numpy array of (I + 1) weights [w0, w1, ..., wI]
    """
    num_inputs = 0 if inputs is None else inputs.shape[1]
    weights = np.zeros(shape=num_inputs + 1)

    """ YOUR CODE HERE """
    if inputs is None:
        weights = np.mean(outputs)
    else:
        tmp_arr = np.ones([outputs.shape[0], 1])
        inputs = np.concatenate([tmp_arr, inputs], axis=1)
        inputs_T = np.transpose(inputs)
        new = np.dot(inputs_T, inputs)
        obs = np.tile(np.array(np.sum(outputs)),new.shape[0])
        weights = np.linalg.solve(new, obs)
    """ END YOUR CODE HERE """
    return weights


def _learn_node_parameter_var(outputs, weights, inputs):
    """
    Returns the variance i.e. sigma^2 for the node. Learns variance for one node only. Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        weights: numpy array of (I + 1) weights of the linear Gaussian model
        inputs:  N x I numpy array of input observations to the linear Gaussian model.

    Returns:
        variance of the node's Linear Gaussian model
    """
    var = 0.
    """ YOUR CODE HERE """
    if inputs is None:
        var = np.dot(outputs-weights,outputs-weights)/outputs.shape[0]
    else:
        tmp_arr = np.ones([outputs.shape[0], 1])
        inputs = np.concatenate([tmp_arr, inputs], axis=1)
        inputs = np.dot(inputs, weights)
        var = np.dot(outputs-inputs,outputs-inputs)/outputs.shape[0]
    """ END YOUR CODE HERE """
    return var


def _get_learned_parameters(nodes, edges, observations):
    """
    Learns the parameters for each node in nodes and returns the parameters as a dictionary. The nodes are given in
    ascending numerical order e.g. [1, 2, ..., V]

    Args:
        nodes: numpy array V nodes in the graph e.g. [1, 2, 3, ..., V]
        edges: numpy array of edges in the graph e.g. [i, j] implies i -> j where i is the parent of j
        observations: dictionary of node: observations pair where observations[1] returns a list of
                    observations for node 1.

    Returns:
        dictionary of parameters e.g.
        parameters = {
            "1": {  // first node
                "bias": w0 weight for node "1",
                "variance": variance for node "1"

                "2": weight for node "2", who is the parent of "1"
                ...
                // weights for other parents of "1"
            },
            ...
            // parameters of other nodes.
        }
    """
    parameters = {}
    """ YOUR CODE HERE """
    for node in nodes:
        outputs = np.array(observations[node])
        parent_node_ls = []
        for edge in edges:
            if node == edge[1]:
                parent_node_ls.append(edge[0])

        if len(parent_node_ls)>0:
            inputs = np.empty([outputs.shape[0],0])
            for parent in parent_node_ls:
                inputs = np.concatenate([inputs, np.array(observations[parent]).reshape([outputs.shape[0], 1])], axis=1)
        else:
            inputs = None

        # learn parameters
        weights = _learn_node_parameter_w(outputs, inputs)
        var = _learn_node_parameter_var(outputs, weights, inputs)
        parameters[node] = {}
        for i, parent in enumerate(parent_node_ls):
            parameters[node][parent] = weights[i+1]
        if len(parent_node_ls)>0:
            parameters[node]['bias'] = weights[0]
        else:
            parameters[node]['bias'] = weights
        parameters[node]['variance'] = var
    """ END YOUR CODE HERE """
    return parameters


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    # argparser = ArgumentParser()
    # argparser.add_argument('--case', type=int, required=True,
    #                        help='case number to create observations e.g. 1 if 1.json')
    # args = argparser.parse_args()

    # case = args.case
    case = '4'
    observation_file = os.path.join(OBSERVATION_DIR, '{}.json'.format(case))
    with open(observation_file, 'r') as f:
         observation_config = json.load(f)

    nodes = observation_config['nodes']
    edges = observation_config['edges']
    observations = observation_config['observations']

    # solution part
    parameters = _get_learned_parameters(nodes=nodes, edges=edges, observations=observations)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    for node, node_params in parameters.items():
        for param, val in node_params.items():
            node_params[param] = float(val)
        parameters[node] = node_params

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(parameters, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()

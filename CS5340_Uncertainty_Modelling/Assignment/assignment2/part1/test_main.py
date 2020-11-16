import json
import os
import unittest

import networkx as nx
import numpy as np

from factor import Factor
from main import (
    DATA_DIR,
    INPUT_DIR,
    _update_mrf_w_evidence,
    get_conditional_probabilities,
    parse_input_file,
)

GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "ground-truth")


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "1.json")
                ),
                "update_mrf_w_evidence": (
                    [0, 1, 2, 3],
                    [[0, 1], [0, 2], [2, 3]],
                    [
                        Factor(var=[0], card=[2], val=[0.6, 0.4]),
                        Factor(
                            var=[0, 1], card=[2, 3], val=[0.6, 0.3, 0.2, 0.7, 0.2, 0.0]
                        ),
                        Factor(var=[0, 2], card=[2, 2], val=[0.15, 0.45, 0.85, 0.55]),
                        Factor(var=[2, 3], card=[2, 2], val=[0.3, 0.5, 0.7, 0.5]),
                    ],
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "1.json")
                ),
            },
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "2.json")
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "2.json")
                ),
            },
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "3.json")
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "3.json")
                ),
            },
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "4.json")
                ),
                "update_mrf_w_evidence": (
                    [0, 1, 2, 3],
                    [[0, 1], [0, 2], [2, 3]],
                    [
                        Factor(var=[0], card=[2], val=[0.6, 0.4]),
                        Factor(
                            var=[0, 1], card=[2, 3], val=[0.6, 0.3, 0.2, 0.7, 0.2, 0.0]
                        ),
                        Factor(var=[0, 2], card=[2, 2], val=[0.15, 0.45, 0.85, 0.55]),
                        Factor(var=[2, 3], card=[2, 2], val=[0.3, 0.5, 0.7, 0.5]),
                    ],
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "4.json")
                ),
            },
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "5.json")
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "5.json")
                ),
            },
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "6.json")
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "6.json")
                ),
            },
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "7.json")
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "7.json")
                ),
            },
            {
                "inputs": parse_input_file(
                    input_file=os.path.join(INPUT_DIR, "8.json")
                ),
                "update_mrf_w_evidence": (
                    [0, 1, 3, 5],
                    [[3, 5]],
                    [
                        Factor(var=[0], card=[2], val=[1, 2]),
                        Factor(var=[1], card=[2], val=[4, 6]),
                        Factor(var=[3], card=[2], val=[10, 8]),
                        Factor(var=[3], card=[2], val=[2, 9]),
                        Factor(var=[3, 5], card=[2, 2], val=[4, 3, 2, 1]),
                        Factor(var=[5], card=[2], val=[2, 8]),
                    ],
                ),
                "get_conditional_probabilities": _read_json(
                    os.path.join(GROUND_TRUTH_DIR, "8.json")
                ),
            },
        ]

    def test_update_mrf_w_evidence(self):
        for i, case in enumerate(self.test_cases):

            if "update_mrt_w_evidence" not in case:
                continue

            inputs = case["inputs"]
            expected = case["update_mrf_w_evidence"]
            nodes, edges, evidence, factors = inputs

            query_nodes, updated_edges, updated_factors = _update_mrf_w_evidence(
                nodes, evidence, edges, factors
            )

            got = nx.Graph()
            got.add_nodes_from(query_nodes)
            got.add_edges_from(updated_edges)

            expected_nodes, expected_edges, expected_factors = expected

            expected = nx.Graph()
            expected.add_nodes_from(expected_nodes)
            expected.add_edges_from(expected_edges)

            self.assertTrue(nx.is_isomorphic(got, expected))
            for factor in updated_factors:
                self.assertTrue(
                    factor in expected_factors,
                    f"case {i}, updated_factor not in expected_factors {factor}",
                )
            for factor in expected_factors:
                self.assertTrue(
                    factor in updated_factors,
                    f"case {i}, expected_factor not in updated_factors",
                )

    def test_get_conditional_probabilities(self):

        for i, case in enumerate(self.test_cases):
            inputs = case["inputs"]
            expected = case["get_conditional_probabilities"]

            nodes, edges, evidence, factors = inputs

            # solution part:
            (
                query_nodes,
                query_conditional_probabilities,
            ) = get_conditional_probabilities(
                all_nodes=nodes, edges=edges, factors=factors, evidence=evidence
            )

            predictions = {}
            for i, node in enumerate(query_nodes):
                probability = query_conditional_probabilities[i].val
                predictions[str(node[0])] = list(np.array(probability, dtype=float))

            for node, pred in expected.items():
                np.testing.assert_allclose(predictions[node], pred)
            # self.assertDictEqual(predictions, expected, f"case {i}")


if __name__ == "__main__":
    unittest.main()

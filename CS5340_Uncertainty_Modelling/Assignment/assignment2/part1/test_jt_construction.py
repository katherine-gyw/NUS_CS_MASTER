import networkx as nx

from factor import Factor
from jt_construction import _get_clique_factors, _get_jt_clique_and_edges
from test_factor_utils import STR_OUTPUT_MISMATCH, wrap_test


@wrap_test
def test_get_clique_factors():

    test_cases = [
        {
            "jt_cliques": [[0, 1], [0, 2], [2, 3]],
            "factors": [
                Factor(var=[0], card=[2], val=[0.6, 0.4]),
                Factor(var=[0, 1], card=[2, 3], val=[0.6, 0.3, 0.2, 0.7, 0.2, 0.0]),
                Factor(var=[0, 2], card=[2, 2], val=[0.15, 0.45, 0.85, 0.55]),
                Factor(var=[2, 3], card=[2, 2], val=[0.3, 0.5, 0.7, 0.5]),
            ],
            "expected_factors": [
                Factor(var=[0, 1], card=[2, 3], val=[0.36, 0.12, 0.12, 0.28, 0.12, 0]),
                Factor(var=[0, 2], card=[2, 2], val=[0.15, 0.45, 0.85, 0.55]),
                Factor(var=[2, 3], card=[2, 2], val=[0.3, 0.5, 0.7, 0.5]),
            ],
        },
    ]

    for i, case in enumerate(test_cases):
        jt_cliques = case["jt_cliques"]
        factors = case["factors"]

        jt_factors = _get_clique_factors(jt_cliques, factors)

        expected_factors = case["expected_factors"]
        for factor in expected_factors:
            assert (
                factor in jt_factors
            ), f"case {i} expected_factor {factor} not in jt_factors, {jt_factors}"

        for factor in jt_factors:
            assert factor in expected_factors


@wrap_test
def test_get_jt_clique_and_edges():

    test_cases = [
        {
            "nodes": [0, 1, 2, 3],
            "edges": [[0, 1], [0, 2], [2, 3]],
            "expected_len": 3,
            "expected_edges": [[0, 1], [1, 2]],
        },
        {
            "nodes": [0, 1, 2, 3],
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "expected_len": 4,
            "expected_edges": [[0, 1], [1, 2], [2, 3]],
        },
    ]

    for case in test_cases:

        nodes = case["nodes"]
        edges = case["edges"]

        jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes, edges)
        got = nx.empty_graph(len(jt_cliques))
        got.add_edges_from(jt_edges)

        expected = nx.empty_graph(case["expected_len"])
        expected.add_edges_from(case["expected_edges"])

        assert nx.is_isomorphic(got, expected)


if __name__ == "__main__":
    test_get_jt_clique_and_edges()
    test_get_clique_factors()

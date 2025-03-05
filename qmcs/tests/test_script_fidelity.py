import random

import networkx as nx
import numpy as np
from ..src.analytical_fidelity_calc import ghz_fidelity_from_star
from ..src.numerical_fidelity_calc import (
    fuse_at_node,
    get_bell_state_dict,
    ghz_fidelity_from_tree,
    reduce_tree,
    remove_qubits,
)
import warnings

try:
    import netsquid as ns
    import netsquid.qubits.operators as ops
    import netsquid.qubits.qubitapi as qapi

    ns.set_qstate_formalism(ns.QFormalism.DM)
    USE_NETSQUID = True
except ImportError:
    warnings.warn(
        "Not using Netsquid, GHZ state fidelity recorded is a approx lower bound term"
    )
    USE_NETSQUID = False


def test_ghz_fidelity_from_star():
    # Create a star graph
    star_graph = nx.Graph()
    source = "source"
    users = ["user1", "user2", "user3"]
    star_graph.add_node(source)
    for user in users:
        star_graph.add_node(user)
        star_graph.add_edge(source, user, w=0.9)

    # Call the function
    f_exact, f_approx, w_r = ghz_fidelity_from_star(star_graph, source, users)

    # Check the results
    assert isinstance(f_exact, float)
    assert isinstance(f_approx, float)
    assert isinstance(w_r, float)
    assert 0 <= f_exact <= 1
    assert 0 <= f_approx <= 1
    assert 0 <= w_r <= 1


def test_ghz_fidelity_from_star_no_users():
    # Create a star graph with no users
    star_graph = nx.Graph()
    source = "source"
    users = []
    star_graph.add_node(source)

    # Call the function
    try:
        ghz_fidelity_from_star(star_graph, source, users)
        assert False
    except Exception as e:
        print(e)
        assert True


def test_ghz_fidelity_from_star_single_user():
    # Create a star graph with a single user
    star_graph = nx.Graph()
    star_graph.add_nodes_from([0, 1])
    w = 0.8
    star_graph.add_edge(0, 1, w=w, w_log=abs(-np.log(w)))
    users = [0]
    # Call the function
    f_exact, f_approx, w_r = ghz_fidelity_from_star(star_graph, 1, users)

    # Check the results
    assert isinstance(f_exact, float)
    assert isinstance(f_approx, float)
    assert isinstance(w_r, float)
    f_expected = (3 * w + 1) / 4

    np.testing.assert_almost_equal(f_exact, f_expected, 5)
    np.testing.assert_almost_equal(f_approx, f_expected, 5)
    np.testing.assert_almost_equal(w_r, w, 5)


def test_ghz_fidelity_from_star_multiple_users():
    # Create a star graph with multiple users
    star_graph = nx.Graph()
    w_init = 0.7

    source = 0
    users = list(range(1, 3))
    for s in users:
        star_graph.add_edge(0, s, w=w_init, w_log=abs(-np.log(w_init)))

    # Call the function
    f_exact, f_approx, w_r = ghz_fidelity_from_star(star_graph, source, users)

    # Check the results
    assert isinstance(f_exact, float)
    assert isinstance(f_approx, float)
    assert isinstance(w_r, float)
    assert 1 >= f_exact
    assert f_exact >= f_approx
    assert f_approx >= w_r
    assert w_r >= 0
    np.testing.assert_almost_equal(f_exact, 0.6175, 3)
    np.testing.assert_almost_equal(f_approx, 0.6, 3)
    np.testing.assert_almost_equal(w_r, 0.49, 3)


def test_ghz_fidelity_star_vs_tree():
    for _ in range(100):
        # Create a star graph with multiple users
        star_graph = nx.Graph()

        source = 0
        users = list(range(1, 5))
        for s in users:
            w_init = 0.5 + random.random() / 2
            star_graph.add_edge(0, s, w=w_init, w_log=abs(-np.log(w_init)))

        # Call the function
        f_exact, f_approx, w_r = ghz_fidelity_from_star(star_graph, source, users)
        f_exact_n, f_approx_n, w_rn = ghz_fidelity_from_tree(star_graph, users)
        if USE_NETSQUID:
            np.testing.assert_almost_equal(f_exact, f_exact_n, 4)
        np.testing.assert_almost_equal(f_approx, f_approx_n, 4)
        np.testing.assert_almost_equal(w_r, w_rn, 4)


def test_ghz_fidelity_simple_tree():
    # Create a tree graph with multiple users
    tree_graph = nx.Graph()
    for u, v in [(0, 1)]:
        tree_graph.add_edge(u, v, w=0.9)
    users = [1, 2]

    # Call the function
    f_exact, f_approx, w_r = ghz_fidelity_from_tree(tree_graph, users)
    assert isinstance(f_exact, float)
    assert isinstance(f_approx, float)
    assert isinstance(w_r, float)
    assert 1 >= f_exact
    assert f_exact >= f_approx
    assert f_approx >= w_r
    assert w_r >= 0
    if USE_NETSQUID:
        np.testing.assert_almost_equal(f_exact, (3 * 0.9 + 1) / 4, 4)
    np.testing.assert_almost_equal(f_approx, (3 * 0.9 + 1) / 4, 4)
    np.testing.assert_almost_equal(w_r, 0.9, 4)


def test_ghz_fidelity_tree2():
    tree_graph = nx.Graph()
    edges = [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 10)]
    users = [1, 2, 7, 10]
    for u, v in edges:
        tree_graph.add_edge(u, v, w=0.9)
    tree_graph = reduce_tree(tree_graph, users)

    # Call the function
    f_exact, f_approx, w_r = ghz_fidelity_from_tree(tree_graph, users)
    assert isinstance(f_exact, float)
    assert isinstance(f_approx, float)
    assert isinstance(w_r, float)
    assert 1 >= f_exact
    assert f_exact >= f_approx
    assert f_approx >= w_r
    assert w_r >= 0
    if USE_NETSQUID:
        np.testing.assert_almost_equal(f_exact, 0.551068114301762, 3)
    np.testing.assert_almost_equal(f_approx, 0.5407311628222657, 3)
    np.testing.assert_almost_equal(w_r, 0.43046721000000004, 3)


def test_ghz_fidelity_tree_n_3():
    # Create a tree graph with multiple users
    tree_graph = nx.Graph()
    edges = [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5)]
    users = [1, 2, 5]
    for u, v in edges:
        tree_graph.add_edge(u, v, w=0.9)
    tree_graph = reduce_tree(tree_graph, users)

    nodes_sorted_degree = list(tree_graph.nodes())
    nodes_sorted_degree.sort(key=lambda n: len(tree_graph.edges(n)), reverse=True)
    nodes_to_operate = [n for n in nodes_sorted_degree if len(tree_graph.edges(n)) > 1]
    if USE_NETSQUID:
        qubit_dict = get_bell_state_dict(tree_graph)
        for node in nodes_to_operate:
            new_ghz, qubit_dict, new_key = fuse_at_node(node, qubit_dict)

        new_ghz, final_key = remove_qubits(new_ghz, new_key, users, qubit_dict)
        assert set(final_key) == set(users)
        assert len(new_ghz) == len(users)
        
    # Call the function

    f_exact, f_approx, w_r = ghz_fidelity_from_tree(tree_graph, users)
    assert isinstance(f_exact, float)
    assert isinstance(f_approx, float)
    assert isinstance(w_r, float)
    assert 1 >= f_exact
    assert f_exact >= f_approx
    assert f_approx >= w_r
    assert w_r >= 0
    if USE_NETSQUID:
        np.testing.assert_almost_equal(f_exact, 0.6855199999999994, 3)
    np.testing.assert_almost_equal(f_approx, 0.6817192187500001, 3)
    np.testing.assert_almost_equal(w_r, 0.5904900000000001, 3)


def test_ghz_fidelity_tree():
    # Create a tree graph with multiple users
    tree_graph = nx.Graph()
    edges = [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    users = [1, 2, 7, 5]
    for u, v in edges:
        tree_graph.add_edge(u, v, w=0.9)
    tree_graph = reduce_tree(tree_graph, users)

    nodes_sorted_degree = list(tree_graph.nodes())
    nodes_sorted_degree.sort(key=lambda n: len(tree_graph.edges(n)), reverse=True)
    nodes_to_operate = [n for n in nodes_sorted_degree if len(tree_graph.edges(n)) > 1]
    if USE_NETSQUID:
        qubit_dict = get_bell_state_dict(tree_graph)
        for node in nodes_to_operate:
            new_ghz, qubit_dict, new_key = fuse_at_node(node, qubit_dict)

        new_ghz, final_key = remove_qubits(new_ghz, new_key, users, qubit_dict)
        assert set(final_key) == set(users)
        assert len(new_ghz) == len(users)
    # Call the function

    f_exact, f_approx, w_r = ghz_fidelity_from_tree(tree_graph, users)
    assert isinstance(f_exact, float)
    assert isinstance(f_approx, float)
    assert isinstance(w_r, float)
    assert 1 >= f_exact
    assert f_exact >= f_approx
    assert f_approx >= w_r
    assert w_r >= 0
    if USE_NETSQUID:
        np.testing.assert_almost_equal(f_exact, 0.5923473250000005, 3)
    np.testing.assert_almost_equal(f_approx, 0.5845742300781251, 3)
    np.testing.assert_almost_equal(w_r, 0.47829690000000014, 3)

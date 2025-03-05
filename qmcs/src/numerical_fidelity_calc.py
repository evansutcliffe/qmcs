import copy
import itertools as it
import warnings

import networkx as nx
import numpy as np

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

# some functions inspired from netsquid code
# https://gitlab.com/softwarequtech/netsquid-snippets/netsquid-factory/


def bsm(bell_a, bell_b, qubit_a=1, qubit_b=0):
    """
    (Func Unused as we use can reduce steiner for Werner parameter / fidelity after bsm)

    Perform a Bell State Measurement (BSM) on two qubits from two Bell pairs.

    This function applies a CNOT gate with the specified qubits as control and target,
    followed by a Hadamard gate on the control qubit. It then measures both qubits in
    the standard basis and applies corrections based on the measurement outcomes.

    Parameters:
    bell_a (list): The first Bell pair.
    bell_b (list): The second Bell pair.
    qubit_a (int, optional): The index of the qubit in bell_a to be used as the control qubit. Default is 1.
    qubit_b (int, optional): The index of the qubit in bell_b to be used as the target qubit. Default is 0.

    Returns:
    list: A new Bell pair formed by the unmeasured qubits from bell_a and bell_b.
    """
    a1 = bell_a[qubit_a]
    a2 = bell_b[qubit_b]
    ns.qubits.operate([a1, a2], ns.CNOT)  # CNOT: a1 = control, a2 = target
    ns.qubits.operate(a1, ns.H)
    # Measure a1 in the standard basis:
    m1, _ = ns.qubits.measure(a1, discard=True)
    # unused output is probability
    m2, _ = ns.qubits.measure(a2, discard=True)
    # unused output is probability
    if m1 == 1:
        ns.qubits.operate(bell_a[0], ns.Z)
    if m2 == 1:
        ns.qubits.operate(bell_b[1], ns.X)
    new_bell = [bell_a[0]] + [bell_b[1]]  # BUG??
    return new_bell


def fuse(ghz_a: list, ghz_b: list, index_a: int, index_b: int):
    """
    Fuse two GHZ states at specified indices.

    This function takes two GHZ states (represented as lists of qubits) and fuses them
    by performing a CNOT operation between the qubits at the specified indices. The qubit
    from the second GHZ state is measured and discarded. If the measurement result is 1,
    an X operation is applied to all remaining qubits in the second GHZ state. The resulting
    fused GHZ state is returned.

    Args:
        ghz_a (list): The first GHZ state, represented as a list of qubits.
        ghz_b (list): The second GHZ state, represented as a list of qubits.
        index_a (int): The index of the qubit in the first GHZ state to be used in the fusion.
        index_b (int): The index of the qubit in the second GHZ state to be used in the fusion.

    Returns:
        list: The fused GHZ state, represented as a list of qubits.
    """
    qubit_a = ghz_a[index_a]
    qubit_b = ghz_b[index_b]
    ns.qubits.operate([qubit_a, qubit_b], ns.CNOT)  # CNOT: a1 = control, a2 = target
    m1, _ = ns.qubits.measure(qubit_b, discard=True)
    ghz_b.pop(index_b)
    if m1 == 1:
        [ns.qubits.operate(q, ns.X) for q in ghz_b]
    ghz_new = ghz_a + ghz_b
    return ghz_new


def fuse_at_node(node, qubit_dict):
    """
    Fuses all GHZ states associated with a given node into a single GHZ state.

    Args:
        node (int): The node at which the GHZ states are to be fused.
        qubit_dict (dict): A dictionary where keys are tuples representing GHZ states
                           and values are the corresponding qubit objects.

    Returns:
        tuple: A tuple containing:
            - new_ghz: The new fused GHZ state.
            - qubit_dict: The updated dictionary with the new fused GHZ state.
            - new_key: The new key (list of nodes) representing the fused GHZ state.
    """
    # tuple name representing each ghz state to be operated in
    ghz_states_shared_with_node = [
        ghz_tuple_name for ghz_tuple_name in qubit_dict.keys() if node in ghz_tuple_name
    ]
    # list of qubits objects
    qubits_to_fuse = [qubit_dict[key] for key in ghz_states_shared_with_node]
    # index qubits of each ghz state held at this node
    qubit_indices = [list(key).index(node) for key in ghz_states_shared_with_node]

    control_index = qubit_indices[0]
    new_key = list(ghz_states_shared_with_node[0])
    new_ghz = qubits_to_fuse[0]  # base of new ghz state
    # Combine all other GHZ states into ghz[0] using a fusion operation
    for ghz_b_state, ghz_b, index_b in list(
        zip(ghz_states_shared_with_node, qubits_to_fuse, qubit_indices)
    )[1:]:
        new_ghz = fuse(new_ghz, ghz_b, index_a=control_index, index_b=index_b)
        new_key.extend(list(ghz_b_state))
    new_key = list(dict.fromkeys(new_key))  # new key (list of nodes ghz state held at)
    # remove combined ghz states from dict
    [qubit_dict.pop(ghz) for ghz in ghz_states_shared_with_node]
    # add new combined ghz to dict
    qubit_dict[tuple(new_key)] = new_ghz
    return new_ghz, qubit_dict, new_key


def remove_qubits(ghz, new_key, users, qubit_dict):
    new_ghz = []
    final_key = []

    qubit_in_ghz_index = [
        i for (i, ghz_node) in enumerate(list(new_key)) if ghz_node in users
    ][0]
    for i, ghz_node in enumerate(list(new_key)):
        qubit = qubit_dict[tuple(new_key)][i]
        if ghz_node not in users:
            m, _ = ns.qubits.measure(qubit, observable=ops.X, discard=True)
            if m == 1:
                ns.qubits.operate(ghz[qubit_in_ghz_index], ns.Z)
        else:
            new_ghz.append(qubit)
            final_key.append(ghz_node)
    return new_ghz, final_key


def ghz_mat(n: int = 2, w: float = 1):
    """
    Generate an N-qubit GHZ (Greenberger-Horne-Zeilinger) state density matrix.

    Parameters:
    n (int): Number of qubits. Must be a positive integer greater than 1. Default is 2.
    w (float): Weighting factor for the GHZ state. Must be a float such that 0 <= w <= 1. Default is 1.

    Returns:
    np.ndarray: A complex-valued density matrix representing the GHZ state.

    Raises:
    ValueError: If `n` is not a positive integer greater than 1.
    ValueError: If `w` is not a float between 0 and 1 (inclusive).
    """
    if not (n > 0 and isinstance(n, int)):
        raise ValueError(f"n ({n}) should be a postive interger > 1 for N-qubit GHZ")
    if not 0 <= w <= 1:
        raise ValueError(f"w ({w}) should be float such that 0<=w<=1")
    dim = 2**n
    ghz_state = np.zeros((dim, dim), dtype=np.complex128)
    ghz_state[0, 0] = 1 / 2
    ghz_state[0, -1] = 1 / 2
    ghz_state[-1, 0] = 1 / 2
    ghz_state[-1, -1] = 1 / 2
    if w < 1:
        max_mixed_state = np.eye(dim) / dim
        ghz_state = w * ghz_state + (1 - w) * max_mixed_state
    return ghz_state


def ghz_werner(n: int = 2, w: float = 1.0):  # direct state
    """
    Generate a multiparite werner-like mixed GHZ-type.
    This is a Werner state for n=2
    rho = w |GHZXGHZ| + (1-w) I/D , d=2^n

    Parameters:
    n (int): The number of qubits. Default is 2.
    w (float): The Werner state parameter. Default is 1.0.

    Returns:
    list: A list of qubits representing the GHZ Werner state.
    """

    ghz_qubits = qapi.create_qubits(n, no_state=True)
    ns.qubits.assign_qstate(ghz_qubits, ghz_mat(n=n, w=w))
    return ghz_qubits


def reduce_tree(graph: nx.Graph, users: list):
    """
    Reduces a given graph by transforming a Steiner tree into a tree with edges
    between users and 'fork' nodes only. This function effectively turns branches
    into edges with a Werner parameter calculated as the product of the parameters
    of the edges in the branch.

    Parameters:
    graph (nx.Graph): The input graph to be reduced.
    users (list): A list of user nodes that should remain in the reduced graph.

    Returns:
    nx.Graph: A reduced copy of the input graph with the specified transformations applied.
    """
    graph_copy = copy.deepcopy(graph)
    for node in [
        n for n in graph_copy.nodes if graph_copy.degree(n) == 2 and n not in users
    ]:
        edges = list(graph_copy.edges(node))
        new_edge = list(
            set(it.chain(*edges)) - set([node])
        )  # get two adjecent nodes to node
        w_swapped = np.prod([graph_copy[u][v]["w"] for (u, v) in edges], dtype=float)
        graph_copy.add_edge(new_edge[0], new_edge[1], w=w_swapped)
        graph_copy.remove_node(node)
    return graph_copy


def ghz_fidelity_from_tree(tree: nx.Graph, users: list):
    """
    Calculate the fidelity of a GHZ state generated from a Steiner tree.

    This function takes a reduced Steiner tree and a list of users, and calculates
    the fidelity of the resulting GHZ state. The function performs fusion operations
    on the nodes of the tree, measures out control qubits if necessary, and computes
    the fidelity of the final GHZ state.

    Parameters:
    reduced_tree (nx.Graph): A NetworkX graph representing the reduced Steiner tree.
    users (list): A list of user nodes in the graph.

    Returns:
    float: The fidelity of the resulting GHZ state.
    """
    # assume tree has already been reduced
    f_approx, w_r = analytical_tree_fidelity(tree, users)
    nodes_sorted_degree = list(tree.nodes())
    nodes_sorted_degree.sort(key=lambda n: len(tree.edges(n)), reverse=True)
    nodes_to_operate = [n for n in nodes_sorted_degree if len(tree.edges(n)) >= 2]
    if not USE_NETSQUID or len(nodes_to_operate) == 0:
        f_exact = f_approx
    else:
        qubit_dict = get_bell_state_dict(tree)
        for node in nodes_to_operate:
            new_ghz, qubit_dict, new_key = fuse_at_node(node, qubit_dict)
        new_ghz, _ = remove_qubits(new_ghz, new_key, users, qubit_dict)
        f_exact = qapi.fidelity(
            qubits=new_ghz, reference_state=ghz_mat(n=len(users)), squared=True
        )
    return f_exact, f_approx, w_r


def get_bell_state_dict(graph: nx.Graph):
    """
    Generate a mapping of graph edges to qubits using the GHZ Werner state.
    e.g. For a tuple key (i,j,k) the list of qubits (Q1,Q2,Q3) represents qubits of a 3-qubit GHZ state held at nodes i,j and k

    Args:
        graph (nx.Graph): A NetworkX graph where each edge has a weight attribute 'w'.

    Returns:
        dict: A dictionary where keys are tuples representing edges (u, v) and values are the qubits of the Werner parameters with value w.
    """
    qubit_mapping = {}
    for u, v, attr in graph.edges(data=True):
        qubit_mapping[(u, v)] = ghz_werner(n=2, w=attr["w"])
    return qubit_mapping


def analytical_tree_fidelity(tree, users):
    """
    Calculate the analytical tree fidelity for a given tree and users.

    This function reduces the input tree based on the provided users and then
    calculates the fidelity approximation and the product of edge weights.

    Parameters:
    tree (networkx.Graph): The input tree graph.
    users (list): A list of users to reduce the tree.

    Returns:
    tuple: A tuple containing:
        - f_approx (float): The fidelity approximation.
        - w_r (float): The product of edge weights in the reduced tree.
    """
    reduced_tree = reduce_tree(tree, users)
    w_arr = [attr["w"] for u, v, attr in reduced_tree.edges(data=True)]
    w_r = np.prod(w_arr)
    f_approx = np.prod([(3 * w + 1) / 4 for w in w_arr])
    return f_approx, w_r

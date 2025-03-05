import networkx as nx
import numpy as np
from networkx.algorithms.flow import shortest_augmenting_path


def analytical_star_fidelity(f_branches: list):
    """
    Calculate the fidelity of a GHZ state given a list of branch fidelities.

    Args:
        f_branches (list): A list of branch fidelities (floats) for each branch.

    Returns:
        float: The calculated fidelity of the GHZ state.

    Notes:
        The function computes the three error terms based on the branch fidelities,
        and then combines these terms to calculate the final fidelity of the GHZ state.
    """
    if len(f_branches) == 1:
        # biparite state is just branch fidelity
        return f_branches[0]
    # Source: Bugholo
    main_term = np.prod(
        [(4 * f_b - 1) / 3 for f_b in f_branches], dtype=float
    )  # prod of w_b
    error_term1 = np.prod(
        [(1 + 2 * f_b) / 3 for f_b in f_branches], dtype=float
    )  # prod of w * 1-w
    error_term2 = np.prod([2 * (1 - f_b) / 3 for f_b in f_branches], dtype=float)
    f_ghz = 0.5 * (main_term + error_term1 + error_term2)
    return f_ghz


def ghz_fidelity_from_star(star_graph: nx.Graph, source, users):
    """
    Calculate the GHZ state fidelity from a star graph.

    Parameters:
    star_graph (nx.Graph): The star graph representing the network.
    source (node): The source node in the star graph.
    users (list of nodes): The list of user nodes to distribute the GHZ state to.

    Returns:
    tuple: A tuple containing:
        - f_exact (float): The exact fidelity of the GHZ state.
        - f_approx (float): The approximate fidelity of the GHZ state.
        - w_r (float): The product of all weights in the disjoint paths.

    Raises:
    ValueError: If no users are provided or if the users are not in the star graph.
    ValueError: If the source is not in the star graph.
    """
    if len(users) == 0 or not set(users) <= set(list(star_graph.nodes)):
        raise ValueError("no users to distribute to", users)

    if source not in star_graph.nodes:
        raise ValueError("source not in graph")

    sink = "super sink"
    assert sink not in star_graph.nodes()

    for user in users:
        star_graph.add_edge(user, sink)

    disjoint_paths = list(
        nx.edge_disjoint_paths(
            star_graph,
            source,
            sink,
            flow_func=shortest_augmenting_path,
            cutoff=len(users),
        )
    )
    # disjoint paths are selected from star using shortest hop distance
    # as least cost star already selected this is fine
    star_graph.remove_node(sink)
    # assert len(disjoint_paths) == len(users)

    disjoint_paths = [path for path in disjoint_paths if len(path) > 2]
    # path of length 2 (e.g. node--super sink) can be ignored as the user is the source
    disjoint_paths = [path[:-1] for path in disjoint_paths]
    # remove super sink from paths

    w_branches = []
    w_r = 1  # prod of all w in R
    for path in disjoint_paths:
        w_branch = np.prod([star_graph[u][v]["w"] for u, v in zip(path, path[1:])])
        w_branches.append(w_branch)  # approx
        w_r *= w_branch

    f_branches = [(w * 3 + 1) / 4 for w in w_branches]  # fidelity of each branch
    f_exact = analytical_star_fidelity(f_branches)
    f_approx = np.prod(f_branches)

    return f_exact, f_approx, w_r

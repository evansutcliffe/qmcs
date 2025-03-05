import time
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
import tqdm
import copy

from .analytical_fidelity_calc import ghz_fidelity_from_star
from .numerical_fidelity_calc import ghz_fidelity_from_tree, reduce_tree
from .plotting_helpers import save_data, random_string
from .simulation_helpers import (
    get_best_source_and_star,
    get_star,
    star_from_dict,
    steiner_tree,
)

sim_rng = np.random.default_rng()


def args_routing(
    graph: nx.Graph, users: list, func_str: str, source_node_first_guess=None
):
    """
    Determines the routing strategy and allocates the appropriate graph
    resources for routing based on the given function string.

    Parameters:
    graph (networkx.Graph): The input graph representing the network.
    users (list): A list of user nodes in the graph.
    func_str (str): A string indicating the routing function to use.
                    It can be one of the following:
                    - "sp-s": Single-path (star) (routing with a source node).
                    - "sp-t": Single-path tree.
                    - "mp-s": Multi-path star routing (with a source node).
                    - "mp-t": Multi-path tree routing .

    Returns:
    tuple: A tuple containing:
        - useable_graph (networkx.Graph): The graph to be used for routing.
        - source_node (optional): The source node for routing (None if not applicable).
    """

    source_node = None  # default (tree has no source node)
    useable_graph = None

    if func_str == "sp-t":
        useable_graph = steiner_tree(
            graph, users, weight="p_edge_log", method="pcst"
        ).copy()

    if func_str in ["sp-s", "mp-s"]:
        source_node, useable_graph, _ = get_best_source_and_star(
            graph,
            users,
            weight="p_edge_log",
            source_node_first_guess=source_node_first_guess,
        )
        # useable_graph for sp-s defines edges in Routing solution which link generation attempted

    if func_str == "mp-s" and (source_node_first_guess is not None):
        # if a first guess is given (e.g. the centroid) for mp-s accept it
        _, _, is_valid = get_star(graph, source_node_first_guess, users, "p_edge_log")
        if is_valid:
            source_node = source_node_first_guess
        # else default is source node same as sp-s

    if func_str in ["mp-s", "mp-t"]:  # mp-s
        # useable graph for mp protocols is whole graph
        useable_graph = graph
        # routing_solution can selected from any edge in graph/ G' for multipath protocols

    return useable_graph, source_node


def simulate_network(args: dict):
    """
    Simulates a network protocol based on the provided arguments.

    Args:
        args (dict): A dictionary containing the following keys:
            - "identifier" (dict): A dictionary with the following keys:
                - "id" (str): A unique identifier used to seed the random number generator.
                - "func" (str): The name of the protocol to simulate.
                    Must be one of ["sp-s", "sp-t", "mp-s", "mp-t"].
            - "rng" (np.random.Generator, optional): A random number generator. If not provided, one will be created using the seed.

    Returns:
        tuple: A tuple containing the results of the simulation.
               The first element is the main result, and the second element is either a dictionary or a list that includes the time taken for the simulation.

    Raises:
        ValueError: If the provided function name is not one of the expected values.
    """
    t0 = time.time()
    seed = int.from_bytes(args["identifier"]["id"].encode(), "little") % 2**16 + 1
    # not amazingly random
    args["rng"] = np.random.default_rng(seed)  # each thread needs its own rng
    func_str = args["identifier"]["func"]  # name of protocol to simulate

    if func_str in ["sp-t", "mp-t"]:
        results = routing_tree(args)
    elif func_str in ["sp-s", "mp-s"]:
        results = routing_star(args)
    else:
        raise ValueError("Unknown function:", func_str)

    results[1]["time_taken"] = time.time() - t0

    return results


def mc_controller(args_list: list, threads: int = 12, show_tqdm=True):
    """
    Controls the execution of the Monte Carlo simulation across multiple threads.

    Args:
        args_list (list): A list of arguments to be passed to the simulate_network function.
        threads (int, optional): The number of threads to use for parallel processing. Defaults to 12.
                                 If None, the number of threads will be set to the number of CPU cores minus 2.

    Returns:
        list: A list of results from the simulate_network function.
    """
    if threads is None:
        threads = cpu_count() - 2

    results = []
    with Pool(threads) as p:
        if show_tqdm:
            for ans in tqdm.tqdm(
                p.imap(simulate_network, args_list),
                total=len(args_list),
                smoothing=50 / len(args_list),
            ):
                results.append(ans)
        else:
            results = list(p.imap(simulate_network, args_list))
    return results


def run_experiment(
    args_list: list,
    sim_params: dict,
    threads: int = 12,
    save_dir: str = None,
):
    """
    Runs a simulation experiment with the given parameters and saves the results.

    Parameters:
    args_list (list): A list of arguments to be passed to the simulation controller.
    sim_params (dict): A dictionary containing simulation parameters.
    experiement_name (str, optional): The name of the experiment. Defaults to "figX".
    str_id (str, optional): A unique string identifier for the experiment. If None, a random 6-character string is generated. Defaults to None.
    threads (int, optional): The number of threads to use for the simulation. Defaults to 12.

    Returns:
    list: The samples generated by the simulation.

    Side Effects:
    Saves the results of the experiment to a file and prints a message indicating the experiment has finished.
    """
    if "str_id" not in sim_params:
        sim_params["str_id"] = random_string()
    str_id = sim_params["str_id"]

    if "experiment_name" not in sim_params:
        sim_params["experiment_name"] = "res"
    print(f"start fig4-{str_id}")

    samples = mc_controller(args_list, threads=threads)
    save_data(samples, args_list, sim_params, save_dir=save_dir)
    print("experiment finished", str_id)
    return samples


def timeslot(vectorised_graph: dict, rng: np.random.Generator = None):
    """
    Simulates the state transition of edges in a graph over a timeslot.
    See timeslot_nx function for more detail

    Parameters:
    g_edge_state (numpy.ndarray): The current state of the edges in the graph.
    p_arr_mat (numpy.ndarray): The probability array matrix for successful generation.
    qc_arr_mat (numpy.ndarray): The quantum correction array matrix to update the edge states.
    rng (numpy.random.Generator, optional): Random number generator. If None, a default generator is used.

    Returns:
    None: The function updates g_edge_state in place.
    """
    if rng is None:
        rng = sim_rng
    vectorised_graph["g_prime"] -= 1
    discard = vectorised_graph["g_prime"] <= 0
    # Discard when q_edge_state <= 0
    # e.g. qc = 1 , store for 1 timeslot, discard next timeslot when g_state = 0
    p_success = (
        rng.random(vectorised_graph["g_prime"].shape) < vectorised_graph["p_array"]
    )
    # successful generation
    new_edge_mask = (discard) & (p_success)
    np.putmask(vectorised_graph["g_prime"], new_edge_mask, vectorised_graph["qc_array"])


def timeslot_nx(
    graph: nx.Graph, graph_prime: nx.Graph, rng: np.random.Generator = None
):
    """
    UNUSED: just to show how non-vectorised timeslot operates
    Simulates the evolution of entanglement links over a network graph.

    This function performs two key operations:
    1. Simulates decoherence of any Bell pairs stored over the network.
       - Uses the variable delta, which is the probability of depolarisation.
       - Iterates down the number of timeslots the edge can be stored from (from qc down to 0).
       - If the link is too old (T_remaining = 0) or below the minimum acceptable w parameter, the link is discarded.
       - Stores the entanglement links as G' (graph_prime).
    2. Attempts new entanglement link generation.
       - This is attempted once per each k channels.
       - Generation is not attempted if an entanglement link (edge in G') already exists between u-v over channel k_i.
       - Generation succeeds with probability 'p_edge' and the link is initialized with Werner parameter w and max age qc defined from G.
       - Removes edges which are below f_min (fidelity or age).

    Parameters:
    graph (nx.Graph): The original network graph.
    graph_prime (nx.Graph): The graph storing entanglement links.
    rng (optional): Random number generator instance. If None, uses sim_rng.

    Returns:
    None
    """
    if rng is None:
        rng = sim_rng
    remove = []
    for u, v, key, link in graph_prime.edges(data=True, keys=True):
        # first simulate decoherence and remove old links
        link["T_remaining"] -= 1
        if link["T_remaining"] == 0:
            remove.append((u, v, key))
        else:
            link["w"] *= graph[u][v]["delta"]  # fix this
            link["w_log"] = -np.log(link["w"])
    [graph_prime.remove_edge(u, v, key=key) for u, v, key in remove]
    # remove edges below decoherence or tslot

    for u, v, attr in graph.edges(data=True):
        # simulate link generation (can store up to k links per edge)
        for k in range(attr["k_edge"]):
            if attr["p_edge"] > rng.random() and not graph_prime.get_edge_data(
                u, v, key=k, default=False
            ):
                graph_prime.add_edge(
                    u,
                    v,
                    w=attr["w_init"],
                    T_remaining=attr["qc"],
                    w_log=-np.log(attr["w_init"]),
                )


def setup_g_prime(graph: nx.Graph):
    """
    Sets up various matrices and a dictionary based on the edges of the input graph G.

    Parameters:
    graph (networkx.Graph): A graph where each edge has attributes 'k_edge', 'p_edge', and 'qc'.

    Returns:
    tuple: A tuple containing:
        - g_edge_state (numpy.ndarray): A zero-initialized array of shape (number of edges, max k_edge).
        - p_arr_mat (numpy.ndarray): An array where each row corresponds to the 'p_edge' values of an edge.
        - qc_arr_mat (numpy.ndarray): An array where each row corresponds to the 'qc' values of an edge.
        - g_edge_dict (dict): A dictionary mapping edge indices to edge data.
    """
    k_multiedge = max(attr["k_edge"] for _, _, attr in graph.edges(data=True))
    # we we k = 1 but code is designed to be ready for multi-edge graphs
    g_edge_state = np.zeros((len(graph.edges), k_multiedge), dtype=int)
    p_arr_mat = np.zeros((len(graph.edges), k_multiedge))
    qc_arr_mat = np.zeros((g_edge_state.shape), dtype=int)
    for i, (_, _, attr) in enumerate(graph.edges(data=True)):
        k = attr["k_edge"]
        p_arr_mat[i, :k] = attr["p_edge"]
        qc_arr_mat[i, :k] = attr["qc"]
    vectorised_graph = {}
    vectorised_graph["g_prime"] = g_edge_state
    vectorised_graph["p_array"] = p_arr_mat
    vectorised_graph["qc_array"] = qc_arr_mat
    vectorised_graph["edge_dict"] = dict(enumerate(graph.edges()))
    return vectorised_graph


def generate_nx_g_prime(graph: nx.Graph, vectorised_graph: dict):
    """
    Generates a new networkx graph graph_prime based on the input graph  and the state of entangled edges.

    Parameters:
    graph (networkx.Graph): The original graph.
    g_edge_dict (dict): A dictionary where keys are edge indices and values are tuples representing the nodes connected by the edge.
    g_edge_state (numpy.ndarray): A 2D array where each row represents an edge and each column represents the state of a Bell pair on that edge.

    Returns:
    networkx.Graph: A new graph graph_prime with nodes from graph and edges based on the entangled edges in g_edge_state.
    """
    entangled_edges_id = np.where(np.any(vectorised_graph["g_prime"] > 0, axis=1))[0]
    graph_prime = nx.Graph()
    graph_prime.add_nodes_from(graph)
    for i in entangled_edges_id:
        [u, v] = vectorised_graph["edge_dict"][i][0:2]
        attr = graph[u][v]
        for j in range(attr["k_edge"]):
            bell_pair_age_remaining = vectorised_graph["g_prime"][i, j]
            if bell_pair_age_remaining > 0:
                w = attr["w_init"] * graph[u][v]["delta"] ** (
                    attr["qc"] - bell_pair_age_remaining
                )
                tau = attr["qc"] - bell_pair_age_remaining  # age
                graph_prime.add_edge(u, v, tau=tau, w=w, w_log=abs(-np.log(w)))
    return graph_prime


def modified_connected_component(graph: nx.Graph, vectorised_graph: dict, source):
    """
    Computes the connected component of a graph starting from a source node, considering only edges that are entangled.
    runs A fast BFS node generator modified from networkx for edge_list input.

    Parameters:
    graph (networkx.Graph): The input graph.
    g_edge_state (np.ndarray): A 2D array where each row represents the state of an edge.
                                An edge is considered entangled if any value in its row is greater than 0.
    g_edge_dict (dict): A dictionary mapping edge indices to tuples of node pairs (u, v).
    source_node (int): The node from which to start the connected component search.

    Returns:
    list: A list of nodes in the connected component containing the source node.
    """
    g_edge_dict = vectorised_graph["edge_dict"]
    entangled_edges_id = np.where(np.any(vectorised_graph["g_prime"] > 0, axis=1))[0]
    # for multi-edge
    edge_list = [
        (g_edge_dict[edge_i][0], g_edge_dict[edge_i][1])
        for edge_i in entangled_edges_id
    ]

    n_degree = 0
    adj = {n: [] for n in graph.nodes()}
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
        if source in [u, v]:
            n_degree += 1
    n = len(adj)
    seen = {source}
    nextlevel = [source]
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen and ((v, w) in edge_list or (w, v) in edge_list):
                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return seen, n_degree
    return seen, n_degree


def routing_star(params: dict):
    """
    Simulates the routing of multipartite entanglement in a quantum network using a star approach.
    This code works for sp-s where usable_graph is routing solution (R)
    and mp-t where usable_graph is (by default) the whole network graph

    Args:
        params (dict): A dictionary containing the following keys:
            - "useable_graph" (networkx.Graph): The graph representing the quantum network.
            - "source_node" (Any): The source node in the network.
            - "users" (list of int): The list of user nodes in the network.
            - "reps" (int): The number of repetitions for the simulation.
            - "timesteps" (int): The maximum number of timesteps for the simulation.
            - "rng" (numpy.random.Generator): The random number generator for the simulation.
            - "f_min" (float): The minimum fidelity threshold for generating GHZ states.
            - "multi" (bool): A flag indicating whether to allow multiple GHZ state generations per rep.
            - "identifier" (str): An identifier for the simulation run.

    Returns:
        tuple: A tuple containing:
            - data (tuple): A tuple containing lists of:
                - multipartite_gen_time (list of int): The generation times of multipartite states.
                - multipartite_fidelity (list of float): The fidelities of the generated multipartite states.
                - tree_sizes (list of int): The sizes of the trees used for GHZ state generation.
                - routing_error (list of float): The routing errors for the generated multipartite states.
                - tau_data (list of float): The average tau values for the edges in the generated trees.
            - metadata (dict): A dict containing:
                - reps (int): The number of repetitions for the simulation.
                - identifier (str): The identifier for the simulation run.
                - timesteps (int): The maximum number of timesteps for the simulation.
    """

    datasets = -1 * np.ones((5, params["reps"]))
    # datasets are gen time, fideity, tree size, f_exact-f_approx,average age
    vectorised_graph = setup_g_prime(params["useable_graph"])
    source_node = params["source_node"]
    destination_nodes = set(params["users"]) - set([source_node])

    for i in range(params["reps"]):
        vectorised_graph["g_prime"] = np.zeros(
            vectorised_graph["g_prime"].shape, dtype=int
        )
        t = 0
        n_ghz = 0
        graph_prime = None
        while t < params["timesteps"] and datasets[0, i] == -1:
            timeslot(vectorised_graph, rng=params["rng"])
            t += 1  #
            connected_component, source_node_degree = modified_connected_component(
                params["useable_graph"], vectorised_graph, source_node
            )
            attempt_ghz = (
                source_node_degree >= len(destination_nodes)
                and set(destination_nodes) <= connected_component
            )
            if attempt_ghz:
                graph_prime = generate_nx_g_prime(
                    params["useable_graph"], vectorised_graph
                )

            while attempt_ghz and graph_prime.degree(params["source_node"]) >= len(
                destination_nodes
            ):
                _, flowdict, is_valid = get_star(
                    graph_prime, source_node, destination_nodes, weight="w_log"
                )
                if is_valid:
                    routing_solution = star_from_dict(graph_prime, flowdict)
                    f_ghz, f_approx, _ = ghz_fidelity_from_star(
                        routing_solution, source_node, destination_nodes
                    )

                    if f_approx >= params["f_min"]:
                        # use f_approx to match tree approach

                        datasets[0][i + n_ghz] = t if n_ghz == 0 else 0
                        datasets[1][i + n_ghz] = f_ghz
                        datasets[2][i + n_ghz] = len(routing_solution.edges())
                        datasets[3][i + n_ghz] = f_ghz - f_approx
                        datasets[4][i + n_ghz] = np.mean(
                            [
                                graph_prime[u][v]["tau"]
                                for u, v in routing_solution.edges()
                            ]
                        )
                        n_ghz += 1

                        graph_prime.remove_edges_from(routing_solution.edges())

                # attempt another ghz if
                # 1) its permitted by the protocol
                # 2) last ghz had a high enough fidelity
                # 3) all users are still connected in g_prime
                attempt_ghz = (
                    params["multi"]
                    and f_approx >= params["f_min"]
                    and set(destination_nodes)
                    <= nx.node_connected_component(graph_prime, params["source_node"])
                )

    data = tuple(list(datasets[i]) for i in range(len(datasets)))
    metadata = {
        "reps": params["reps"],
        "identifier": params["identifier"],
        "timesteps": params["timesteps"],
    }
    return data, metadata


def routing_tree(params: dict):
    """
    Simulates the generation of multipartite GHZ distribution in a quantum network a tree approach.
    This code works for sp-t where usable_graph is routing solution (R)
    and mp-s where usable_graph is (by default) the whole network graph
    Args:
        params (dict): A dictionary containing the following keys:
            - "useable_graph" (networkx.Graph): The graph representing the quantum network.
            - "users" (list): List of user nodes in the network.
            - "reps" (int): Number of repetitions for the simulation.
            - "timesteps" (int): Maximum number of timesteps for the simulation.
            - "rng" (numpy.random.Generator): Random number generator for stochastic processes.
            - "f_min" (float): Minimum fidelity threshold for GHZ state generation.
            - "multi" (bool): Flag indicating whether multiple GHZ states can be generated.
    Returns:
        tuple: A tuple containing:
            - data (list): A list of lists containing the following elements:
                - multipartite_gen_time (list): Generation times for multipartite states.
                - multipartite_fidelity (list): Fidelities of the generated multipartite states.
                - tree_sizes (list): Sizes of the routing trees used for GHZ state generation.
                - routing_error (list): Errors in the routing process.
                - tau_data (list): Average tau values for the routing trees.
            - metadata (dict): A dict containing metadata about the simulation:
                - Number of repetitions.
                - Identifier for the simulation.
                - Number of timesteps.
    """
    datasets = -1 * np.ones((5, params["reps"]))
    # datasets are gen time, fideity, tree size, f_exact-f_approx,average age
    vectorised_graph = setup_g_prime(params["useable_graph"])
    users = params["users"]
    for i in range(params["reps"]):
        vectorised_graph["g_prime"] = np.zeros(
            vectorised_graph["g_prime"].shape, dtype=int
        )

        t = 0
        n_ghz = 0
        graph_prime = None
        while t < params["timesteps"] and datasets[0][i] == -1:
            timeslot(vectorised_graph, rng=params["rng"])
            t += 1
            connected_component, _ = modified_connected_component(
                params["useable_graph"],
                vectorised_graph,
                users[0],
            )
            attempt_ghz = set(users) <= connected_component
            if attempt_ghz:
                # convert from vec to networkx graph
                graph_prime = generate_nx_g_prime(
                    params["useable_graph"], vectorised_graph
                )

            while attempt_ghz:
                routing_solution = steiner_tree(
                    graph_prime.subgraph(connected_component),
                    users=users,
                    weight="w_log",
                    method="pcst",
                )

                reduced_tree = reduce_tree(routing_solution.copy(), users)
                f_approx = np.prod(
                    [
                        (3 * attr["w"] + 1) / 4
                        for _, _, attr in reduced_tree.edges(data=True)
                    ]
                )

                if f_approx >= params["f_min"]:
                    f_ghz, f_approx, _ = ghz_fidelity_from_tree(reduced_tree, users)
                    datasets[0][i + n_ghz] = t if n_ghz == 0 else 0
                    datasets[1][i + n_ghz] = f_ghz
                    datasets[2][i + n_ghz] = len(routing_solution.edges())

                    datasets[3][i + n_ghz] = f_ghz - f_approx
                    datasets[4][i + n_ghz] = np.mean(
                        [graph_prime[u][v]["tau"] for u, v in routing_solution.edges()]
                    )
                    n_ghz += 1

                    graph_prime.remove_edges_from(routing_solution.edges())
                    connected_component = nx.node_connected_component(
                        graph_prime, users[0]
                    )

                attempt_ghz = (
                    params["multi"]
                    and f_approx >= params["f_min"]
                    and set(users) <= connected_component
                )

    data = tuple(list(datasets[i]) for i in range(len(datasets)))
    metadata = {
        "reps": params["reps"],
        "identifier": params["identifier"],
        "timesteps": params["timesteps"],
    }
    return data, metadata

import random
import string

import networkx as nx
import numpy as np
import pcst_fast
from networkx.algorithms.approximation.steinertree import (
    steiner_tree as steiner_tree_nx,
)
from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
from networkx.algorithms.flow import cost_of_flow, min_cost_flow


def min_edge_weight(graph: nx.Graph, weight: str, postive_only: bool = False):
    if postive_only:
        return min(
            attr[weight] for _, _, attr in graph.edges(data=True) if attr[weight] > 0
        )
    return min(attr[weight] for _, _, attr in graph.edges(data=True))


def set_default_args(useable_graph, users, source_node, sim_params, func_str):
    """
    Initializes and returns a dictionary of default arguments for a simulation.

    Parameters:
    useable_graph (object): The graph structure to be used in the simulation.
    users (list): A list of users involved in the simulation.
    source_node (int): The source node in the graph.
    sim_params (dict): A dictionary containing simulation parameters such as 'timesteps' and 'reps'.
    func_str (str): A string representing the function identifier.

    Returns:
    dict: A dictionary containing the default arguments for the simulation.
    """
    args = {}
    # protocol vars
    args["useable_graph"] = useable_graph
    args["users"] = users
    args["source_node"] = source_node
    args["multi"] = False
    args["f_min"] = 0.0  # min accepted GHZ state fidelity

    # simulations vars
    args["timesteps"] = sim_params["timesteps"]
    args["reps"] = sim_params["reps"]
    args["rng"] = None

    # identification vars
    args["identifier"] = {}
    args["identifier"]["func"] = func_str
    args["identifier"]["id"] = "".join(
        random.choice(string.ascii_lowercase + string.ascii_uppercase) for _ in range(6)
    )
    return args


def add_default_network_attributes(graph: nx.Graph):
    """
    Adds default network attributes to the given graph.

    This function sets the following default attributes:
    - p_edge: Probability of an link being generated (default is 0.5)
    - qc: memory cutoff time in timeslots  (default is 1)
    - k_edge: number of edges per edge in G (k>1 is a multigraph) (default is 1)
    - w_init: Initial Werner parameter (w_0) (default is for f = 0.99)
    - delta: Decoherence constant (default is 0.99)
    - length: physical length in km -  unused but can be used to define p_edge in terms of loss (default is 1)

    Args:
        graph: The graph to which the attributes will be added.

    Returns:
        The graph with the added default attributes.
    """
    f = 0.99  # fidelity
    w = (4 * f - 1) / 3  # F = w + (1-w)/2^2 => w = (4f-1)/3
    update_graph_params(
        graph, p_edge=0.5, qc=1, k_edge=1, w_init=w, delta=0.99, length=1
    )
    # update_graph_attr(graph, params)


def grid_network(n: int, m: int):
    """
    Creates a 2D grid graph and adds default network attributes.

    Parameters:
    n (int): The number of rows in the grid.
    m (int): The number of columns in the grid.

    Returns:
    networkx.Graph: A 2D grid graph with default network attributes.
    """
    graph = nx.grid_2d_graph(n, m)
    add_default_network_attributes(graph)
    return graph


def update_graph_params(graph: nx.Graph, **kwargs):
    """
    Update the attributes of the edges in the given graph with the provided keyword arguments.

    Parameters:
    graph (networkx.Graph): The graph whose edge attributes are to be updated.
    **kwargs: Arbitrary keyword arguments representing the edge attributes to be updated.
                Each key-value pair in kwargs corresponds to an edge attribute name and its value.

    Notes:
    - The function sets the edge attributes in the graph using the provided key-value pairs.
    - If the key is "p_edge" or "w_init", the function also adds a new attribute with the log of the value.
        The new attribute name will be the original key with "_log" appended to it.
    - Be careful with typos in the keyword arguments, as they will directly affect the edge attributes.
        e.g "qc" != "Qc" so it won't update properly
    """
    for key, value in kwargs.items():
        nx.set_edge_attributes(graph, value, key)
        if key in ["p_edge", "w_init"]:
            nx.set_edge_attributes(graph, abs(-np.log(value)), f"{key}_log")


def centroid_grid(users: list):
    """
    Calculate the centroid of a grid based on user coordinates.

    Args:
        users (list of tuple): A list of tuples where each tuple contains the (x, y) coordinates of a user.

    Returns:
        tuple: A tuple containing the (x, y) coordinates of the centroid.
    """
    x = int(np.round(np.mean([x for x, y in users])))
    y = int(np.round(np.mean([y for x, y in users])))
    return (x, y)


def make_star_from_paths(graph: nx.Graph, best_paths: list):
    """
    Create a new graph with the same nodes as the input graph, but only with the edges
    specified in the best_paths.

    Parameters:
    graph (nx.Graph): The original graph from which nodes and edge data are taken.
    best_paths (list of lists): A list of paths, where each path is a list of nodes
                                representing the best paths to be included in the new graph.

    Returns:
    nx.Graph: A new graph containing the same nodes as the input graph and edges from the best_paths,
              with edge attributes copied from the original graph.
    """
    star = graph.__class__()
    # star.add_nodes_from(graph.nodes(data=True))
    for path in best_paths:
        for u, v in zip(path, path[1:]):
            star.add_edge(u, v)
            star.edges[u, v].update(graph.get_edge_data(u, v))
            star.nodes[u].update(graph.get_node_data(u))
            star.nodes[v].update(graph.get_node_data(v))

    return star


def make_auxillary_graph(
    graph: nx.Graph, users: list, weight: str, source, aux_graph: nx.Graph | None = None
):
    """
    Creates an auxiliary graph for edge connectivity analysis.

    This function takes an input graph and modifies it to create an auxiliary graph
    that includes a super sink node connected to all user nodes. The weights of the
    edges are scaled, and the auxiliary graph is built to facilitate edge connectivity
    analysis.

    Parameters:
    graph (networkx.Graph): The input graph.
    users (list): A list of user nodes to be connected to the super sink.
    weight (str): The attribute name for edge weights in the input graph.
    source (str): The source node in the graph.

    Returns:
    networkx.Graph: The modified auxiliary graph with added super sink and scaled weights.
    """

    # smallest edge has interger weight 100
    if aux_graph is None:
        aux_graph = build_auxiliary_edge_connectivity(graph)
        scaling = 100 / min_edge_weight(graph, weight, postive_only=True)

        for u, v, e in graph.edges(data=True):

            aux_graph[u][v]["weight"] = int(e[weight] * scaling)
            aux_graph[v][u]["weight"] = int(e[weight] * scaling)
            # can be set for multigraph capacity but not needed for current work
            aux_graph[u][v]["capacity"] = 1
            aux_graph[v][u]["capacity"] = 1

    assert "super_sink" not in graph.nodes
    aux_graph.add_node("super_sink")
    for user in users:
        aux_graph.add_edge(user, "super_sink", weight=0, capacity=1)
        aux_graph.add_edge("super_sink", user, weight=0, capacity=1)
    # graph.remove_node(super_sink)
    aux_graph.nodes["super_sink"]["demand"] = 1 * len(users)
    aux_graph.nodes[source]["demand"] = -1 * len(users)
    return aux_graph


def get_star(graph: nx.Graph, source, users, weight="p_edge_log", aux_graph=None):
    """
    Constructs an auxiliary graph with a super sink node and calculates the minimum cost flow
    from the source to the super sink.

    Parameters:
    graph (nx.Graph): The input graph.
    source (node): The source node in the graph.
    users (list): List of user nodes to connect to the super sink.
    weight (str): The edge attribute to use as weight. Default is "p_edge_log".

    Returns:
    tuple: A tuple containing:
        - cost (float): The cost of the minimum cost flow.
        - flowdict (dict): A dictionary representing the flow on each edge.
        - success (bool): True if the flow was successfully calculated, False otherwise.
    """

    aux_graph = make_auxillary_graph(graph, users, weight, source)

    try:
        flowdict = min_cost_flow(aux_graph)
        cost = cost_of_flow(aux_graph, flowdict, weight="weight")
    except nx.NetworkXUnfeasible:
        cost = np.inf
        flowdict = None
    # aux_graph.remove_node("super_sink")
    # aux_graph.nodes[source]["demand"] = 0
    return cost, flowdict, bool(cost < np.inf)


def star_from_dict(graph: nx.Graph, flowdict: dict):
    """
    Create a new graph (star) from an existing graph and a flow dictionary.

    This function constructs a new graph by adding nodes from the original graph
    and edges based on the flow dictionary. Only edges with a positive flow value
    and not connected to a "super_sink" node are added to the new graph. The edge
    attributes from the original graph are preserved.

    Parameters:
    graph (nx.Graph): The original graph from which nodes and edge attributes are copied.
    flowdict (dict): A dictionary representing the flow values between nodes.

    Returns:
    nx.Graph: A new graph with nodes from the original graph and edges based on the flow dictionary.
    """
    star = graph.__class__()
    star.add_nodes_from(graph.nodes(data=True))
    for node_u, value in flowdict.items():
        for node_v, v in value.items():
            if v > 0 and "super_sink" not in [node_u, node_v]:
                star.add_edge(node_u, node_v)
                star.edges[node_u, node_v].update(graph.get_edge_data(node_u, node_v))
    return star


def get_best_source_and_star(
    graph: nx.Graph, users: list, weight, source_node_first_guess=None
):
    """
    Determines the best source node and corresponding star subgraph in a given graph based on the specified weight.

    Parameters:
    graph (nx.Graph): The input graph.
    users (list): A list of user nodes.
    weight (str): The edge attribute to be used as weight.
    first_guess (optional): An optional initial guess for the best source node.

    Returns:
    tuple: A tuple containing the best source node, the best star subgraph, and the best cost.

    Raises:
    ValueError: If no valid star subgraph is found.
    """
    best_source = None
    best_cost = np.inf
    best_flowdict = None
    # #aux_graph = make_auxillary_graph(
    #     graph, users=[], weight=weight, source=list(graph.nodes())[0]
    # )
    for source in list([source_node_first_guess] + list(graph.nodes())):
        cost, flowdict, is_valid = get_star(graph, source, users, weight)
        if is_valid and cost < best_cost:
            best_cost = cost
            best_source = source
            best_flowdict = flowdict

    if best_flowdict is None:
        raise ValueError("No valid star found")

    best_star = star_from_dict(graph, best_flowdict)
    scaling = 100 / min_edge_weight(graph, weight, postive_only=True)
    # zero cost edges exist e.g. if Fidelity = 1 then w_log = abs(-log(w)) = 0
    best_cost /= scaling

    return best_source, best_star, best_cost


def steiner_tree(graph: nx.Graph, users, weight: str = "w_log", method="mehlhorn"):
    """
    Compute the Steiner tree for a given graph and set of users.

    Parameters:
    graph (nx.Graph): The input graph.
    users (list): List of user nodes for which the Steiner tree is computed.
    weight (str): The edge attribute to be used as weight. Default is "w_log".
    method (str): The method to compute the Steiner tree. Options are "pcst" and any from networkx, Default is "mehlhorn".

    Returns:
    nx.Graph: The computed Steiner tree.
    """
    if method == "pcst":  # requires pcst fast and numpy <2
        tree = pcst_steiner(graph, users, weight)
    else:
        tree = steiner_tree_nx(graph, users, weight, method)
    return tree


def pcst_steiner(graph: nx.Graph, users: list, weight: str = "w_log"):
    """
    # mapping of pcst_fast (Prize-Collecting Steiner Tree) to networkx graph
    We do not care about prize so set weight is set uniform and arbitarily high prize

    First we check if the graph is a multi-graph, and if it is we take the least cost subgraph of 'graph'
    and then runs the PCST algorithm to find a subgraph that connects a given set of user nodes
    with minimal cost.

    Parameters:
    graph (nx.Graph): The input graph.
    users (list): A list of user nodes that need to be connected.
    weight (str, optional): The edge attribute to be used as weight. Default is "w_log".

    Returns:
    nx.Graph: A subgraph of the original graph that connects the user nodes with minimal cost.
    """
    graph_best_edges = get_best_edge_subgraph(
        graph, graph.edges(), weight
    )  # get best link from multi-graph
    int_node_mapping = dict(enumerate(graph_best_edges.nodes()))
    node_int_mapping = {n: i for i, n in enumerate(graph_best_edges.nodes())}

    node_id = list(int_node_mapping)
    edge_list = []
    costs = []
    has_zero_cost = min_edge_weight(graph, weight) == 0
    # if an edge exists with zero cost, add small factor to all edge weights
    for u, v, attr in graph_best_edges.edges(data=True):
        edge_weight = attr[weight] + has_zero_cost * 1e-16
        edge_list.append([node_int_mapping[u], node_int_mapping[v]])
        costs.append(edge_weight)

    prizes = [0] * len(node_id)
    for s in users:
        prizes[node_int_mapping[s]] = 99999  # some high value to catch no matter what
    root = -1  # no root
    num_clusters = 1
    # unused variable vertices
    _, eg = pcst_fast.pcst_fast(
        edge_list, prizes, costs, root, num_clusters, "strong", 0
    )

    edges = (
        (int_node_mapping[p[0]], int_node_mapping[p[1]])
        for p in [edge_list[k] for k in eg]
    )
    steiner_tree_of_graph = get_best_edge_subgraph(graph, edges, weight)
    # get best link from multi-graph
    return steiner_tree_of_graph


def get_best_edge_subgraph(graph, edges, weight):
    """
    Returns a subgraph containing the best edges based on the given weight.

    Parameters:
    graph (networkx.Graph): The input graph, which can be a multi-graph.
    edges (iterable): An iterable of edge tuples (u, v) to consider.
    weight (str): The edge attribute to use for determining the best edge.

    Returns:
    networkx.Graph: A subgraph containing the best edges based on the given weight.
    """
    if graph.is_multigraph():
        # min cost edge from multiedge graph
        edges = (
            (u, v, min(graph[u][v], key=lambda k: graph[u][v][k][weight]))
            for u, v in edges
        )
    return graph.edge_subgraph(edges)

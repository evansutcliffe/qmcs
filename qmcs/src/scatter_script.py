import copy
import random

import numpy as np
import tqdm

from .plotting_helpers import enumerated_product
from .simulate import args_routing
from .simulation_helpers import (
    centroid_grid,
    grid_network,
    set_default_args,
    update_graph_params,
)


def setup_args_scatter(sim_params=None):
    """
    Sets up the arguments for the scatter simulation.
    This function initializes the simulation parameters, generates a list of users if not provided,
    and creates a list of argument dictionaries for each combination of functions, users, and parameter ranges.
    Args:
        sim_params (dict, optional): A dictionary containing simulation parameters. If None, default parameters are used.
    Returns:
        tuple: A tuple containing:
            - args_list (list): A list of dictionaries, each containing the arguments for a single simulation run.
            - sim_params (dict): The updated simulation parameters.
    """
    if sim_params is None:  # paper values
        sim_params = default_params()

    if "users_list" not in sim_params:
        sim_params["users_list"] = [
            random.sample(list(sim_params["graph"].nodes()), sim_params["|S|"])
            for _ in range(sim_params["S_reps"])
        ]

    args_list = []
    for (_, j), (func, users) in tqdm.tqdm(
        enumerated_product(sim_params["funcs"], sim_params["users_list"])
    ):
        useable_graph, source_node = args_routing(
            graph=sim_params["graph"],
            users=users,
            func_str=func,
            source_node_first_guess=centroid_grid(users),
        )
        
        for (_, _), (p, qc) in enumerated_product(
            sim_params["p_range"], sim_params["qc_range"]
        ):
            useable_graph_i = copy.deepcopy(useable_graph)
            update_graph_params(useable_graph_i, p_edge=p, qc=qc)
            # update edge params of graph
            # generic args (can overwrite if required)
            args = set_default_args(
                useable_graph_i, users, source_node, sim_params, func
            )
            # non generic args
            args["identifier"]["qc"] = qc  # qc
            args["identifier"]["user"] = j  # user index
            args_list.append(args)
    print("setup finished")
    return args_list, sim_params


def default_params():
    """
    Generates a dictionary of default simulation parameters.

    Returns:
        dict: A dictionary containing the following keys:
            - "timesteps" (int): Number of timesteps for the simulation.
            - "reps" (int): Number of repetitions for the simulation.
            - "|S|" (int): Size of the set S.
            - "S_reps" (int): Number of repetitions for set S.
            - "p_range" (list): List containing the range of probabilities.
            - "funcs" (list): List of function names to be used in the simulation.
            - "qc_range" (numpy.ndarray): Array containing the range of qc values.
            - "graph" (networkx.Graph): A graph object representing the network.
    """
    sim_params = {"timesteps": 10000, "reps": 300, "|S|": 4, "S_reps": 60}
    sim_params["p_range"] = [0.1]
    sim_params["funcs"] = ["sp-s", "sp-t", "mp-s", "mp-t"]
    sim_params["qc_range"] = np.arange(1, 21)
    sim_params["graph"] = grid_network(6, 6)
    return sim_params

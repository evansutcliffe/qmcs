import networkx as nx
import numpy as np
import tqdm

from .plotting_helpers import (
    calculate_distribution_rate,
    enumerated_product,
    save_data,
    random_string,
)
from .simulate import args_routing, run_experiment, simulate_network
from .simulation_helpers import (
    centroid_grid,
    grid_network,
    set_default_args,
    update_graph_params,
)


def default_params():
    """
    Returns the default simulation parameters for the distance script.
    Note that average_f_min is for average GHZ state fidelity in over each sim run, (not each ghz 'f_min')

    The parameters include:
    - timesteps: Number of timesteps in the simulation (default: 20).
    - reps: Number of repetitions for the simulation (default: 300).
    - |S|: Size of the set S (default: 4).
    - S_reps: Number of repetitions for set S (default: 1).
    - p_edge: Probability of an edge in the simulation (default: 0.3).
    - funcs: List of functions to be used in the simulation (default: ["sp-s", "sp-t", "mp-s", "mp-t"]).
    - distance: Range of distances to be considered in the simulation (default: np.arange(3, 10)).
    - average_f_min: Minimum average fidelity for the GHZ state in the protocol (default: 2/3).
    - qc_max: Maximum quantum capacity (default: 20).

    Returns:
        dict: A dictionary containing the default simulation parameters.
    """
    sim_params = {"timesteps": 10000, "reps": 300, "|S|": 4, "S_reps": 1}
    sim_params["p_edge"] = 0.3
    sim_params["funcs"] = ["sp-s", "sp-t", "mp-s", "mp-t"]
    sim_params["distance"] = np.arange(3, 10)
    sim_params["average_f_min"] = 2 / 3
    sim_params["qc_max"] = 20
    return sim_params


def best_cutoff(
    graph: nx.Graph,
    d: int,
    users: list,
    sim_params: dict,
    func_str: str,
    qc_range: list,
):
    """
    Determines the best cutoff value for a given graph and simulation parameters.

    Parameters:
    graph (nx.Graph): The input graph representing the network.
    d (int): The distance parameter for the simulation.
    users (list): List of users in the network.
    sim_params (dict): Dictionary containing simulation parameters.
    func_str (str): String representing the function to be used for routing.
    qc_range (list): List of possible cutoff values to be tested.

    Returns:
    tuple: A tuple containing the best sample, the corresponding arguments, and the best cutoff value.
    """
    first_guess = centroid_grid(users)
    useable_graph, source_node = args_routing(graph, users, func_str, first_guess)
    best_rate = 0
    b_sample = None
    qc_best = None
    b_args = None
    reattempt = True
    for qc in qc_range:
        if reattempt:
            update_graph_params(useable_graph, qc=qc)
            args = set_default_args(
                useable_graph, users, source_node, sim_params, func_str
            )
            args["threshold"] = 0  # hard limit on GHZ fidelity for each state generated
            args["identifier"]["distance"] = d
            args["identifier"]["qc"] = qc
            args["identifier"]["user"] = 0
            sample = simulate_network(args)
            rate, fidelity, n_success, _ = calculate_distribution_rate(sample)

            if rate > best_rate and fidelity >= sim_params["average_f_min"]:
                if rate > best_rate * 2 or qc_best is None:
                    qc_best = qc
                b_sample = sample
                b_args = args
                best_rate = rate

            reattempt = not (n_success == 0 or rate < best_rate / 2)

    return b_sample, b_args, qc_best


def run_distance_results(sim_params: dict = None, nice_plots: bool = False):
    """
    Run distance results simulation and save the data.

    Parameters:
    fig_str (str): The figure string identifier.
    experiement_name (str, optional): The name of the experiment. Defaults to "figd".
    sim_params (dict, optional): The simulation parameters. If None, default parameters will be used. Defaults to None.
    nice_plots (bool, optional): If True, rerun data for more timesteps for best qc values. Defaults to False.

    Returns:
    None
    """

    if sim_params is None:
        sim_params = default_params()
    if "str_id" not in sim_params:
        sim_params["str_id"] = random_string()
    str_id = sim_params["str_id"]

    if "experiment_name" not in sim_params:
        sim_params["experiment_name"] = "distance"

    print(f"start figd-{str_id}")

    samples = []
    b_args_list = []
    qc_matrix = sim_params["qc_max"] * np.ones(
        (len(sim_params["funcs"]), len(sim_params["distance"]))
    )
    qc_starting = sim_params["qc_max"] * np.ones(qc_matrix.shape)

    for (i, k), (func_str, d) in tqdm.tqdm(
        enumerated_product(sim_params["funcs"], sim_params["distance"]),
        total=np.prod(qc_matrix.size),
    ):
        graph = grid_network(d, d)
        update_graph_params(graph, p_edge=sim_params["p_edge"])

        users = [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]
        if qc_starting[i, k - 1] != 0:
            qc_range = np.arange(qc_starting[i, k - 1], 0, -1)
            b_sample, b_args, qc_best = best_cutoff(
                graph, d, users, sim_params, func_str, qc_range
            )

            if b_sample is None:  # give up
                qc_matrix[i, k] = 0
            else:
                samples.append(b_sample)
                b_args_list.append(b_args)
                qc_starting[i, k] = qc_best
        else:
            qc_matrix[i, k] = 0

    if nice_plots:
        # rerun data for more timesteps for best qc values
        sim_params["timesteps"] = 10000
        samples = run_experiment(b_args_list, sim_params)
    save_data(samples, b_args_list, sim_params)

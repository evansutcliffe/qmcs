import itertools as it
import math
import os
import pickle
import random
import string

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def random_string(n: int = 6):
    """
    Generate a random string of lowercase letters.

    Parameters:
    n (int): The length of the random string to generate. Default is 6.

    Returns:
    str: A random string of lowercase letters of length n.
    """
    return "".join(random.choices(string.ascii_lowercase, k=n))


def enumerated_product(*args):  # cleaner for multiple for loops
    """
    Generates an enumerated Cartesian product of input iterables.

    Args:
        *args: Variable length argument list of iterables.

    Yields:
        tuple: A tuple containing two elements:
            - A tuple of indices corresponding to the position of each element in the Cartesian product.
            - A tuple of elements from the Cartesian product of the input iterables.

    Example:
        >>> list(enumerated_product('AB', 'CD'))
        [((0, 0), ('A', 'C')), ((0, 1), ('A', 'D')), ((1, 0), ('B', 'C')), ((1, 1), ('B', 'D'))]
    """
    yield from zip(it.product(*(range(len(x)) for x in args)), it.product(*args))


def print_params(str_id: str = None, res: dict = None):
    """
    Print the parameters from the result dictionary and load from file if data if not provided.
    Args:
        str_id (str, optional): The string identifier to load data if `res` is not provided. Defaults to None.
        res (dict, optional): The result dictionary containing parameters. If not provided, data will be loaded using `str_id`. Defaults to None.
    Raises:
        ValueError: If both `res` and `str_id` are not provided.
    Prints:
        The key-value pairs of the parameters.
    """
    if res is None:
        if str_id is None:
            raise ValueError("no data or valid string_id provided")
        res = load_data(str_id)

    for key, value in res["params"].items():
        if key == "users_list" and len(value) > 1:
            print("len users list", len(res["params"]["users_list"]))
        else:
            print(key, value)


def save_data(samples, args_list, sim_params, save_dir=None):
    """
    Save simulation results, arguments, and parameters to pickle files.

    Parameters:
    str_id (str): A unique identifier for the simulation run.
    experiment_name (str): The name of the experiment.
    samples (any): The simulation samples to be saved.
    args_list (list): The list of arguments used in the simulation.
    sim_params (dict): The simulation parameters.

    Returns:
    None
    """
    str_id = sim_params["str_id"]
    experiment_name = sim_params["experiment_name"]
    if save_dir is None:
        save_dir = os.getcwd() + "//results"
    with open(f"{save_dir}//{experiment_name}_{str_id}_samples.pickle", "wb") as file:
        pickle.dump(samples, file)

    with open(f"{save_dir}//{experiment_name}_{str_id}_args.pickle", "wb") as file:
        pickle.dump(args_list, file)

    with open(f"{save_dir}//{experiment_name}_{str_id}_params.pickle", "wb") as file:
        pickle.dump(sim_params, file)


def load_data(name_id, my_dir=None, not_args=False):
    """
    Load data from pickle files in the specified directory that match the given name identifier.

    Args:
        name_id (str): The identifier to match in the filenames.
        my_dir (str, optional): The directory to search for files. Defaults to "..//results".

    Returns:
        dict: A dictionary where the keys are the object names extracted from the filenames and the values are the loaded objects.

    Raises:
        ValueError: If no files matching the name_id are found in the specified directory.
    """
    if my_dir is None:
        my_dir = os.getcwd() + "//results"
    files = [f for f in os.listdir(my_dir) if name_id in f and f.endswith(".pickle")]
    if not_args:
        files = [f for f in files if not f.endswith("args.pickle")]
    if len(files) == 0:
        raise ValueError(f"dataset {name_id} not found")

    res = {}
    for filename in files:
        with open(f"{my_dir}//{filename}", "rb") as file:
            load_object = pickle.load(file)
        ob_name = filename.split(".")[0].split("_")[-1]
        res[ob_name] = load_object
    return res


def delete_data(name_id, my_dir=None):
    """
    Deletes files in the specified directory that contain the given name_id and have a .pickle extension.

    Args:
        name_id (str): The identifier to look for in filenames.
        my_dir (str, optional): The directory to search for files. Defaults to "..//results".

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If the file cannot be deleted due to permission issues.
    """
    if my_dir is None:
        my_dir = os.getcwd() + "//results"
    directory = os.listdir(my_dir)
    for filename in directory:
        if name_id in filename and filename.endswith(".pickle"):
            os.remove(f"{my_dir}//{filename}")


def param_to_index(param: str):
    """
    Convert a parameter name to its corresponding index.

    Parameters:
    param (str): The name of the parameter to convert. Valid values are 'rate',
                 'multipartite_gen_time', 'multipartite_fidelity', 'tree_sizes',
                 'routing_error', and 'tau_data'.

    Returns:
    int: The index corresponding to the parameter name.

    Raises:
    ValueError: If the parameter name is not valid.
    """
    if param == "rate":
        return 0
    data_types = [
        "multipartite_gen_time",
        "multipartite_fidelity",
        "tree_sizes",
        "routing_error",
        "tau_data",
    ]
    if param not in data_types:
        raise ValueError(f"not valid param type ({param})")
    return data_types.index(param)


def filter_samples(samples: list, to_match: dict):
    """
    Filters a list of samples based on matching criteria provided in a dictionary.

    Args:
        samples (list): A list of samples where each sample is expected to be a tuple,
                        and the second element of the tuple is a list or dictionary
                        containing the sample's attributes.
        to_match (dict): A dictionary containing key-value pairs to match against
                         the sample's attributes.

    Returns:
        list: A list of samples that match all the criteria specified in the to_match dictionary.
    """
    sub_samples = []
    for sample in samples:
        _, metadata = sample
        if isinstance(metadata, list):
            sample_id = metadata[1]
        else:
            sample_id = metadata["identifier"]
        if all(sample_id[key] == value for key, value in to_match.items()):
            sub_samples.append(sample)
    return sub_samples


def calculate_distribution_rate(sample: tuple):
    """
    Calculate the distribution rate and fidelity from the given sample data.

    Parameters:
    sample (tuple): A tuple containing sample data and sample metadata.
        - sample_data (list): A list where the first element is an array of generation times and the second element is an array of fidelities.
        - sample_metadata (list): A list where the third element is either a dictionary containing simulation parameters or another type.
    sim_params (dict): A dictionary containing simulation parameters, including the number of timesteps.

    Returns:
    tuple: A tuple containing:
        - rate (float): The calculated distribution rate.
        - fidelity (float or None): The mean fidelity of valid fidelities, or None if no valid fidelities are present.
        - n_success (int): The number of successful runs.
    """
    sample_data, sample_metadata = sample
    gen_time_arr = np.array(sample_data[0])
    max_timesteps_per_run = sample_metadata["timesteps"]
    successful_runs = gen_time_arr[gen_time_arr >= 0]
    n_success = len(successful_runs)
    n_runs = len(gen_time_arr)
    t_total = sum(successful_runs) + (n_runs - n_success) * max_timesteps_per_run
    rate = n_success / t_total

    fidelity_arr = np.array(sample_data[1])
    valid_fidelity_arr = fidelity_arr[fidelity_arr != -1.0]
    if len(valid_fidelity_arr) > 0:
        fidelity = np.mean(valid_fidelity_arr)
    else:
        fidelity = None
    return rate, fidelity, n_success, t_total


def get_sample_data(sample, param_index):
    """
    Extracts and processes sample data based on the given parameter index.

    Parameters:
    sample (tuple): A tuple containing data and metadata.
                    - data: A list or array-like structure containing sample data.
                    - metadata: A dictionary containing metadata information.
    param_index (int): The index of the parameter to extract from the data.

    Returns:
    tuple: A tuple containing:
            - valid_data (numpy.ndarray): An array of valid data points (excluding -1.0).
            - identifier (str): The identifier extracted from the metadata.
    """
    data, metadata = sample
    identifier = metadata["identifier"]
    data_arr = np.array(data[param_index])
    valid_data = data_arr[data_arr != -1.0]
    return valid_data, identifier


def get_index_of_sample(
    sim_params: dict, variables: list, keys: list, identifier: dict
):
    """
    Get the index of a sample based on simulation parameters and identifiers.

    Args:
        sim_params (dict): A dictionary containing simulation parameters.
        variables (list): A list of variable names to look up in sim_params.
        keys (list): A list of keys to look up in the identifier.
        identifier (dict): A dictionary containing identifiers for the sample.

    Returns:
        tuple: A tuple containing the indices of the sample in sim_params and the user identifier.
    """
    indexs = [
        list(sim_params[var]).index(identifier[key])
        for var, key in zip(variables, keys)
    ]
    indexs.append(identifier["user"])
    return tuple(indexs)


def setup_datasets(sim_params: dict, variables: list):
    """
    Initializes and sets up data matrices for simulation parameters and variables.

    Args:
        sim_params (dict): A dictionary containing simulation parameters.
        variables (list): A list of variable names to be used for setting up the datasets.

    Returns:
        tuple: A tuple containing three numpy arrays:
            - data_matrix (numpy.ndarray): A zero-initialized array with dimensions based on the lengths of the variables and 'S_reps' in sim_params.
            - sample_count_matrix (numpy.ndarray): A zero-initialized array with the same shape as data_matrix to count samples.
            - valid_ghzs_matrix (numpy.ndarray): A zero-initialized array with the same shape as data_matrix to count valid GHZ states.
    """
    matrix_vars = [sim_params[v] for v in variables]
    matrix_vars.append(list(range(sim_params["S_reps"])))
    data_matrix = np.zeros(tuple(len(x) for x in matrix_vars))
    sample_count_matrix = np.zeros(data_matrix.shape)
    valid_ghzs_matrix = np.zeros(data_matrix.shape)
    return data_matrix, sample_count_matrix, valid_ghzs_matrix


def dataset_to_matrix(
    samples: list,
    sim_params: dict,
    variables: list,
    keys: list,
    var_param="rate",
    min_ghzs: int = 200,
    remove_empty_data: bool = True,
):
    """
    Converts a dataset of samples into a matrix format based on simulation parameters and variables.

    Parameters:
    -----------
    samples : list
        List of sample data to be processed.
    sim_params : dict
        Dictionary containing simulation parameters.
    variables : list
        List of variables to be considered in the dataset.
    keys : list
        List of keys to identify samples.
    var_param : str, optional
        Parameter to be used for the variable (default is 'rate').
    min_ghzs : int, optional
        Minimum number of GHZ states required to consider a data point valid (default is 200).
    remove_empty_data : bool, optional
        Flag to indicate whether to remove data points with no valid data (default is True).

    Returns:
    --------
    numpy.ndarray
        Processed dataset in matrix format.
    """
    param_index = param_to_index(var_param)
    dataset, sample_counts, valid_ghzs = setup_datasets(sim_params, variables)
    for sample in samples:
        valid_data, identifier = get_sample_data(sample, param_index)
        indexs = get_index_of_sample(sim_params, variables, keys, identifier)
        if len(valid_data) > 0:
            _, _, n_success, t_slots = calculate_distribution_rate(sample)
            valid_ghzs[indexs] += n_success
            sample_counts[indexs] += 1
            if var_param == "rate":
                dataset[indexs] += t_slots
            else:
                dataset[indexs] += np.mean(valid_data)
        elif remove_empty_data:
            # discard all data for this datapoint as one users as 0 GHZ states
            # empty datapoints skew the average fidelity, path length etc.
            # as further apart /lower fidelity user sets are more likely to have 0 GHZ states
            dataset[indexs] = -1.0 * np.inf

    # remove datapoints with no samples
    dataset[sample_counts == 0] = -1.0 * np.inf
    # divide by datapoints for each sample
    if var_param == "rate":
        # sum n_ghz / sum timeslots
        dataset = valid_ghzs / dataset
    else:
        # Each datapoint is averaged over multiple samples (e.g.g different sets of users S)
        dataset[sample_counts >= 1] /= sample_counts[sample_counts >= 1]
    users_axis = len(dataset.shape) - 1
    dataset = dataset.mean(axis=users_axis)
    valid_ghzs = valid_ghzs.mean(axis=users_axis)
    dataset[valid_ghzs < min_ghzs] = -1.0 * np.inf
    return dataset


def get_xy_data(
    res: dict,
    func: str,
    x_param="rate",
    y_param="multipartite_fidelity",
    variables=["qc_range"],
    keys=["qc"],
    min_ghzs=60,
    remove_empty_data=False,
):
    """
    Extracts and returns x and y data matrices from the given results based on specified parameters.
    Args:
        res (dict): The results dictionary containing samples and parameters.
        func (str): The function name to filter samples.
        x_param (str, optional): The parameter to be used for the x-axis. Defaults to "rate".
        y_param (str, optional): The parameter to be used for the y-axis. Defaults to "multipartite_fidelity".
        variables (list, optional): List of variables to consider. Defaults to ["qc_range"].
        keys (list, optional): List of keys to consider. Defaults to ["qc"].
        min_ghzs (int, optional): Minimum number of GHZ states required. Defaults to 60.
        remove_empty_data (bool, optional): Flag to remove empty data entries. Defaults to False.
    Returns:
        tuple: A tuple containing two matrices, x_matrix and y_matrix, corresponding to the x and y data respectively.
    """

    sub_samples = filter_samples(res["samples"], to_match={"func": func})
    x_matrix = dataset_to_matrix(
        sub_samples,
        sim_params=res["params"],
        variables=variables,
        keys=keys,
        var_param=x_param,
        min_ghzs=min_ghzs,
        remove_empty_data=remove_empty_data,
    )

    y_matrix = dataset_to_matrix(
        sub_samples,
        sim_params=res["params"],
        variables=variables,
        keys=keys,
        var_param=y_param,
        min_ghzs=min_ghzs,
        remove_empty_data=remove_empty_data,
    )
    return x_matrix, y_matrix


def plot_graph_routing_solution(
    graph: nx.Graph,
    routing_solution: nx.Graph,
    users: list,
    source=None,
    weight: str = "p_edge_log",
):
    """
    Plots a graph with a routing solution overlay.

    Parameters:
    graph (nx.Graph): The original graph to be plotted.
    routing_solution (nx.Graph): The graph representing the routing solution.
    users (list): List of user nodes.
    source: The source node.
    weight (str, optional): The edge attribute to use for edge weights. Default is "p_edge_log".

    Returns:
    None
    """
    color_state_map = {1: "green", 0: "lightgrey"}
    node_color = [color_state_map[node in users] for node in graph.nodes()]
    if source is not None:
        node_color[list(graph.nodes()).index(source)] = "red"
    edge_color = []
    weights = []
    for u, v, e in graph.edges(data=True):
        if routing_solution.has_edge(u, v):
            edge_color.append("red")
        else:
            edge_color.append("blue")

        if weight in e:
            weights.append(e[weight] * 5)
        else:
            weights.append(5)
    pos = {(x, y): (y, -x) for x, y in graph.nodes()}

    nx.draw(
        graph,
        pos=pos,
        edge_color=edge_color,
        style="--",
        node_color=node_color,
        with_labels=False,
        node_size=1600,
        width=weights,
    )
    plt.show()


def error_bars(
    samples: list,
    sim_params: dict,
    variables: list,
    keys: list,
    desired_ratio_vs_max_likelihood: float = 1e-3,
):
    """
    Calculate the lower and upper error bars for a given set of samples and simulation parameters.
    Args:
        samples (list): A list of tuples where each tuple contains data and metadata.
                        The data is a dictionary with parameter values, and metadata is a list with identifiers.
        sim_params (dict): A dictionary containing simulation parameters.
        variables (list): A list of variable names to consider for the error bars.
        keys (list): A list of keys corresponding to the variables in the metadata.
        desired_ratio_vs_max_likelihood (float, optional): The desired ratio versus maximum likelihood for error calculation.
                                                            Defaults to 1e-3.
    Returns:
        tuple: Two numpy arrays representing the lower and upper error bars respectively.
    Raises:
        ValueError: If samples represents multiple different user combinations error bars aren't valid
    """

    param_index = param_to_index("multipartite_gen_time")
    matrix_vars = [sim_params[v] for v in variables]
    lower_matrix = np.zeros(tuple(len(x) for x in matrix_vars))
    upper_matrix = np.zeros(lower_matrix.shape)
    count_matrix = np.zeros(lower_matrix.shape)
    for sample in samples:
        data, metadata = sample
        if isinstance(metadata, list):
            identifier = metadata[1]
        else:
            identifier = metadata["identifier"]
        data_arr = np.array(data[param_index])
        valid_data = data_arr[data_arr != -1.0]
        for indexs, var in enumerated_product(*(matrix_vars)):  # get index of each item
            indexs = indexs[0] if len(indexs) == 1 else indexs
            if all(var[i] == identifier[keys[i]] for i in range(len(matrix_vars))):
                # if index match sample dataset
                count_matrix[indexs] += 1
                if len(valid_data) == 0:
                    lower_matrix[indexs] = 0
                    upper_matrix[indexs] = np.inf
                else:
                    n_ghz = len(valid_data)
                    n_attempts = int(
                        np.sum(valid_data)
                        + (len(data_arr) - n_ghz) * sim_params["timesteps"]
                    )
                    lower, upper = likely_error_rate_bounds(
                        n_attempts,
                        n_ghz,
                        desired_ratio_vs_max_likelihood=desired_ratio_vs_max_likelihood,
                    )
                    lower_matrix[indexs] = lower
                    upper_matrix[indexs] = upper
    if not np.all(
        count_matrix <= 1
    ):  # for valid error bars only one user set should be used
        raise ValueError("Errors bars not accurate for different user combinations")

    return lower_matrix, upper_matrix


####################################

# below are likelyhood error rate functions from stim
# https://github.com/Strilanc/honeycomb_threshold/blob/main/src/collect_data.py#L288
#


def log_factorial(n: int) -> float:
    r"""Approximates $\ln(n!)$; the natural logarithm of a factorial.

    Uses Stirling's approximation for large n.
    """
    if n < 20:
        return sum(math.log(k) for k in range(1, n + 1))
    return (n + 0.5) * math.log(n) - n + math.log(2 * np.pi) / 2


def log_binomial(*, p, n: int, hits: int):
    r"""Approximates $\ln(P(hits = B(n, p)))$; the natural logarithm of a binomial distribution.

    All computations are done in log space to ensure intermediate values can be represented as
    floating point numbers without underflowing to 0 or overflowing to infinity. This is necessary
    when computing likelihoods over many samples. For example, if 80% of a million samples are hits,
    the maximum likelihood estimate is p=0.8. But even this optimal estimate assigns a prior
    probability of roughly 10^-217322 for seeing *exactly* 80% hits out of a million (whereas the
    smallest representable double is roughly 10^-324).

    This method can be broadcast over multiple hypothesis probabilities.

    Args:
        p: The independent probability of a hit occurring for each sample. This can also be an array
            of probabilities, in which case the function is broadcast over the array.
        n: The number of samples that were taken.
        hits: The number of hits that were observed amongst the samples that were taken.

    Returns:
        $\ln(P(hits = B(n, p)))$
    """
    # Clamp probabilities into the valid [0, 1] range (in case float error put them outside it).
    p_clipped = np.clip(p, 0, 1)

    result = np.zeros(shape=p_clipped.shape, dtype=np.float32)
    misses = n - hits

    # Handle p=0 and p=1 cases separately, to avoid arithmetic warnings.
    if hits:
        result[p_clipped == 0] = -np.inf
    if misses:
        result[p_clipped == 1] = -np.inf

    # Multiply p**hits and (1-p)**misses onto the total, in log space.
    result[p_clipped != 0] += np.log(p_clipped[p_clipped != 0]) * hits
    result[p_clipped != 1] += np.log1p(-p_clipped[p_clipped != 1]) * misses

    # Multiply (n choose hits) onto the total, in log space.
    log_n_choose_hits = log_factorial(n) - log_factorial(misses) - log_factorial(hits)
    result += log_n_choose_hits

    return result


def binary_search(func, min_x: int, max_x: int, target: float) -> int:
    """Performs an approximate granular binary search over a monotonically ascending function."""
    while max_x > min_x + 1:
        med_x = (min_x + max_x) // 2
        out = func(med_x)
        if out < target:
            min_x = med_x
        elif out > target:
            max_x = med_x
        else:
            return med_x
    fmax = func(max_x)
    fmin = func(min_x)
    dmax = 0 if fmax == target else fmax - target
    dmin = 0 if fmin == target else fmin - target
    return max_x if abs(dmax) < abs(dmin) else min_x


def likely_error_rate_bounds(
    num_samples: int, num_ghz: int, desired_ratio_vs_max_likelihood: float
):
    """Compute relative-likelihood bounds.

    Returns the min/max error rates whose Bayes factors are within the given ratio of the maximum
    likelihood estimate.
    """
    log_max_likelihood = log_binomial(
        p=num_ghz / num_samples, n=num_samples, hits=num_ghz
    )
    target_log_likelihood = log_max_likelihood + math.log(
        desired_ratio_vs_max_likelihood
    )
    acc = 100
    low = (
        binary_search(
            func=lambda exp_err: log_binomial(
                p=exp_err / (acc * num_samples), n=num_samples, hits=num_ghz
            ),
            target=target_log_likelihood,
            min_x=0,
            max_x=num_ghz * acc,
        )
        / acc
    )
    high = (
        binary_search(
            func=lambda exp_err: -log_binomial(
                p=exp_err / (acc * num_samples), n=num_samples, hits=num_ghz
            ),
            target=-target_log_likelihood,
            min_x=num_ghz * acc,
            max_x=num_samples * acc,
        )
        / acc
    )
    return low / num_samples, high / num_samples

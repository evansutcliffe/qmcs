import os

import matplotlib.pyplot as plt
import numpy as np

from .plotting_helpers import (
    dataset_to_matrix,
    error_bars,
    filter_samples,
    get_xy_data,
    load_data,
)

fontsize = 12
names = ["sp-s", "sp-t", "mp-s", "mp-t"]

colour_scheme = [
    "blue",
    "orange",
    "purple",
    "red",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


def add_qc_annotate(x_matrix, y_matrix, sim_params, qc_range=None):
    if qc_range is None:
        qc_range = [1, 2, 5, 10, 20]
    qc_range.append(np.argmax(np.array(x_matrix) > 0) + 1)  # min_qc_show
    for qc in list(set(qc_range)):
        index = np.where(np.array(sim_params["qc_range"]) == qc)[0]
        if len(index) > 0:  # datapoint exists
            x = x_matrix.ravel()[index[0]]  # dim = 1
            y = y_matrix.ravel()[index[0]]  # dim = 1
            text_offset = (x * 1.02, y * 1.01)
            plt.annotate(qc, xy=(x, y), xytext=tuple(text_offset))


def plot_scatter_other(
    fig_str: str,
    x_param: str = "rate",
    y_param: str = "multipartite_fidelity",
    fig_axis: dict = None,
    save_figure: bool = True,
    save_dir: str = None,
    annotate_qc: bool = True,
    f_min_average: float = 0,
    min_ghzs: int = 1,
):
    res = load_data(fig_str)
    sim_params = res["params"]
    plt.figure(figsize=(6, 4))
    for j, func in enumerate(sim_params["funcs"]):
        rate_matrix, fidelity_matrix = get_xy_data(
            res,
            func,
            x_param=x_param,
            y_param=y_param,
            min_ghzs=min_ghzs,
            remove_empty_data=True,
        )
        rate_matrix_good = rate_matrix[fidelity_matrix > f_min_average]
        fidelity_matrix_good = fidelity_matrix[fidelity_matrix > f_min_average]

        plt.scatter(
            rate_matrix_good,
            fidelity_matrix_good,
            label=names[j],
            color=colour_scheme[j],
        )

        if annotate_qc:
            add_qc_annotate(rate_matrix, fidelity_matrix, sim_params)

    plt.xscale("log")
    plt.legend()
    plt.xlabel(x_param, fontsize=fontsize)
    plt.ylabel(y_param, fontsize=fontsize)
    plt.grid(True, linewidth=0.5, which="both")
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if fig_axis is not None:
        plt.xlim(fig_axis["x"])
        plt.ylim(fig_axis["y"])

    if save_dir is None:
        save_dir = os.getcwd()

    if save_figure:
        p = res["params"]["p_range"][0]
        plt.savefig(
            f"{save_dir}//{fig_str}_{x_param}_{y_param}_{p}.pdf", bbox_inches="tight"
        )


def plot_scatter_data(
    fig_str: str,
    fig_axis: dict = None,
    save_figure: bool = True,
    save_dir: str = None,
    annotate_qc: bool = True,
    f_min_average: float = 0,
    min_ghzs: int = 1,
):
    """
    Plots scatter data for multipartite fidelity against distribution rate.

    Parameters:
    fig_str (str): The figure string identifier used for loading data and saving the figure.
    fig_axis (dict, optional): Dictionary specifying the axis limits with keys 'x' and 'y'. Defaults to None.
    save_figure (bool, optional): Flag to save the figure. Defaults to True.
    save_dir (str, optional): Directory to save the figure. Defaults to None, which sets it to the current working directory + "/figures".
    annotate_qc (bool, optional): Flag to annotate the quantum channels. Defaults to True.
    f_min_average (float, optional): Minimum average fidelity to filter data. Must be between 0 and 1. Defaults to 0.
    min_ghzs (int, optional): Minimum number of GHZ states to filter data. Defaults to 1.

    Raises:
    ValueError: If f_min_average is not between 0 and 1.

    Returns:
    None
    """
    if f_min_average < 0 or f_min_average > 1:
        raise ValueError("invalid f_min_average")

    res = load_data(fig_str)
    sim_params = res["params"]
    plt.figure(figsize=(6, 4))
    for j, func in enumerate(sim_params["funcs"]):
        rate_matrix, fidelity_matrix = get_xy_data(
            res,
            func,
            x_param="rate",
            y_param="multipartite_fidelity",
            min_ghzs=min_ghzs,
            remove_empty_data=True,
        )

        rate_matrix_good = rate_matrix[fidelity_matrix > f_min_average]
        fidelity_matrix_good = fidelity_matrix[fidelity_matrix > f_min_average]

        plt.scatter(
            rate_matrix_good,
            fidelity_matrix_good,
            label=names[j],
            color=colour_scheme[j],
        )
        if annotate_qc:
            add_qc_annotate(rate_matrix, fidelity_matrix, sim_params)

    plt.xscale("log")
    plt.legend()
    plt.xlabel(r"Distribution rate (GHZ/$T_{slot}$)", fontsize=fontsize)
    plt.ylabel("GHZ state fidelity", fontsize=fontsize)
    plt.grid(True, linewidth=0.5, which="both")
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if fig_axis is not None:
        plt.xlim(fig_axis["x"])
        plt.ylim(fig_axis["y"])

    if save_dir is None:
        save_dir = os.getcwd() + "//figures"
    if save_figure:
        p = res["params"]["p_range"][0]
        plt.savefig(f"{save_dir}//{fig_str}_rate_fidelity_{p}.pdf", bbox_inches="tight")


def plot_distance_data(
    fig_str: str,
    save_figure: bool = True,
    fig_axis: dict = None,
    save_dir=None,
    min_ghzs: int = 1,
    likelyhood_ratio=1e-3,
):
    """
    Plots distance data from simulation results and saves or displays the figure.

    Parameters:
    fig_str (str): The filename string for the figure.
    save_figure (bool): If True, saves the figure to a file. If False, displays the figure. Default is True.
    fig_axis (dict): Dictionary containing axis limits for the plot. Should have keys 'x' and 'y' with corresponding limits. Default is None.
    save_dir (str): Directory where the figure will be saved. If None, saves to the current working directory under a 'figures' folder. Default is None.
    min_ghzs (int): Minimum GHZs value for filtering the rate matrix. Default is 1.
    likelyhood_ratio (float): Desired ratio versus maximum likelihood for error bars. Default is 1e-3.

    Returns:
    None
    """
    res = load_data(fig_str)
    sim_params = res["params"]
    plt.figure(figsize=(6, 4))
    plt.minorticks_on()
    plt.grid(True, linewidth=0.5, axis="x", which="major")
    plt.grid(True, linewidth=0.5, axis="y", which="both")
    for j, func in enumerate(sim_params["funcs"]):
        sub_samples = filter_samples(res["samples"], to_match={"func": func})
        rate_matrix = dataset_to_matrix(
            sub_samples,
            sim_params,
            variables=["distance"],
            keys=["distance"],
            var_param="rate",
            min_ghzs=min_ghzs,
        )
        valid_x = rate_matrix[rate_matrix > 0]
        valid_y = sim_params["distance"][rate_matrix > 0]
        plt.plot(valid_y, valid_x, label=func, color=colour_scheme[j])
        data_lower, data_upper = error_bars(
            sub_samples,
            sim_params,
            variables=["distance"],
            keys=["distance"],
            desired_ratio_vs_max_likelihood=likelyhood_ratio,
        )
        valid_upper = data_upper[rate_matrix > 0]
        valid_lower = data_lower[rate_matrix > 0]
        plt.fill_between(
            valid_y, valid_lower, valid_upper, alpha=0.3, color=colour_scheme[j]
        )

    plt.legend(fontsize=fontsize)
    plt.xlabel(r"Network grid size $M \times M$", fontsize=fontsize)
    plt.xlim([min(valid_y), max(valid_y)])
    plt.xticks(sim_params["distance"], fontsize=fontsize)

    plt.ylabel(r"Distribution rate (GHZ/$T_{slot}$)", fontsize=fontsize)
    plt.yscale("log")
    plt.yticks(fontsize=fontsize)

    if fig_axis is not None:
        plt.xlim(fig_axis["x"])
        plt.ylim(fig_axis["y"])

    if save_dir is None:
        save_dir = os.getcwd() + "//figures"
    if save_figure:
        plt.savefig(f"{save_dir}//{fig_str}_distance.pdf", bbox_inches="tight")
    else:
        plt.show()
    del res

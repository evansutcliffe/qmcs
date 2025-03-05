import random

from qmcs.src.scatter_script import setup_args_scatter, default_params
from qmcs.src.simulate import mc_controller
from qmcs.src.plotting_helpers import random_string


def test_sim():
    str_id = random_string()
    print(f"start fig-{str_id}")
    sim_params = default_params()
    sim_params["p_range"] = [0.3]
    sim_params["experiment_name"] = "scatter_01"
    sim_params["timesteps"] = 100
    sim_params["S_reps"] = 1
    sim_params["users_list"] = [[(0, 0), (0, 1), (1, 1), (1, 0)]]
    sim_params["str_id"] = str_id
    args_list, sim_params = setup_args_scatter(sim_params=sim_params)

    random.shuffle(args_list)
    # randomise order just to makes timing estimate more accurate
    samples = mc_controller(args_list)
    assert len(samples) == len(args_list)
    return True

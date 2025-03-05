import random

from qmcs.src.plotting_helpers import random_string
from qmcs.src.plotting_script import plot_scatter_data
from qmcs.src.scatter_script import default_params, setup_args_scatter
from qmcs.src.simulate import run_experiment

if __name__ == "__main__":
    str_id = random_string()
    print(f"start fig-{str_id}")
    sim_params = default_params()
    sim_params['p_range']  = [0.2]
    sim_params['experiment_name'] = "scatter_01"
    sim_params["timesteps"] =   10
    sim_params["S_reps"] =   6
    sim_params['str_id']  = str_id
    args_list, sim_params = setup_args_scatter(sim_params=sim_params)

    random.shuffle(args_list) 
     # randomise order just to makes timing estimate more accurate
    run_experiment(args_list, sim_params)
    plot_scatter_data(fig_str=sim_params['str_id'] , save_figure=True)

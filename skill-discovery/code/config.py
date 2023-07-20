import argparse

arg_lists = []
parser = argparse.ArgumentParser(description="skill-discovery")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ("true", "1")

exp_arg = add_argument_group("Experiment Params")

### basics
exp_arg.add_argument(
    "--gpu_id", type=str, default="0", help="GPU id."
)
exp_arg.add_argument(
    "--use_wandb", type=str2bool, default=True, help="Whether to use wandb. "
)
exp_arg.add_argument(
    "--plot_state_occupancies_freq", type=int, default=0, help="Frequency of iterations to plot state occupancies. 0 means no ploting. If it is not zero, it should be dividable by steps_per_call. "
)

### random seeds
exp_arg.add_argument(
    "--train_seed", type=int, default=1, help="The random seed for training."
)

### environment
exp_arg.add_argument(
    "--environment_type", type=str, default="small", choices=['small', 'large'], help="Use a small or large four-room environment."
)
exp_arg.add_argument(
    "--train_goal_duration", type=int, default=8, choices=[8, 20], help="The number of actions to take in the environment per goal period."
)
exp_arg.add_argument(
    "--train_code_arity", type=int, default=24, choices=[24, 128], help="The number of codes/skills to select. Values of codes lie in the range [0, code_arity)."
) 
exp_arg.add_argument(
    "--train_iterations", type=int, default=1000000, help="Training iterations." # 32M for the large environment
)

### methods
exp_arg.add_argument(
    "--reward_type", type=str, default="log_clipping", help="Types of reward functions."
)
exp_arg.add_argument(
    "--train_bonus_weight", type=float, default=20, help="Weight to assign to bonus Q function when selecting actions greedily."
) 
exp_arg.add_argument(
    "--disable_ensemble", type=str2bool, default=False, help="Disable ensemble."
)

### Learning rate
exp_arg.add_argument(
    "--learning_rate", type=float, default=0.004, help="Learning rate."
) 

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
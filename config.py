import argparse

arg_lists = []
parser = argparse.ArgumentParser(description="RAM and DTRAM")

def str2bool(v):
    return v.lower() in ("true", "1")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Special params
exp_arg = add_argument_group("Model Params")
exp_arg.add_argument(
    "--dynamic_exploration_time", type=str2bool, default=True, help="Whether to make the model terminate exploration before reaching the maximum number of steps"
)
exp_arg.add_argument("--device_id", type=int, default=0, help="GPU index for experiments")
exp_arg.add_argument(
    "--use_wandb", type=str2bool, default=True, help="Whether to use wandb for logging"
)
exp_arg.add_argument(
    "--dataset",
    type=str,
    default="cluttered-mnist-60x60-4-8x8",
    help="Select a dataset",
)
exp_arg.add_argument(
    "--random_baseline", type=str2bool, default=False, help="Wether randomly select actions. "
) 

# Dynamic exploration time models related params
exp_arg.add_argument(
    "--use_implict_lat_penalty", type=str2bool, default=True, help="Use a discount factor to penalize latent prediction. "
)
exp_arg.add_argument(
    "--discount_factor_lat_penalty", type=float, default=0.99, help="Contol the penalty intensity for latent prediction. "
)
exp_arg.add_argument(
    "--weight_r_lat", type=float, default=0.1, help="Weight of the latent reward r_lat"
)
exp_arg.add_argument(
    "--only_r_lat_on_success", type=str2bool, default=False, help="Only consider the reward of latency when the prediction is correct"
)
exp_arg.add_argument(
    "--use_acc_cert_weighted_r_lat", type=str2bool, default=False, help="Use accuracy or certainty weighted reward of latency"
)

# Accuracy-based reward related params
exp_arg.add_argument(
    "--acc_reward_type", type=str, default="zero_and_one", help="[minus_one_and_one, zero_and_one]"
)

# Information maximization-based reward related params
exp_arg.add_argument(
    "--use_information_maximization_reward", type=str2bool, default=False, help="Whether to use information maximization-based reward setup instead of accuracy-based reward setup"
)
exp_arg.add_argument(
    "--information_maximization_reward_type", type=str, default="linear_clipping", help="What type of the information maximization-based reward to use."
)

# Resampling. Not used in the paper
exp_arg.add_argument(
    "--data_uncertainty_based_resampling", type=str2bool, default=False, help="Wether we do resampling in evaluation. "
)
exp_arg.add_argument(
    "--data_uncertainty_based_resampling_depth", type=int, default=1, help="Resampling depth"
)

# Ensemble related params. Not used in the paper.
exp_arg.add_argument(
    "--n_classifiers", type=int, default=1, help="Number of classifiers in the ensemble"
)
exp_arg.add_argument(
    "--subset_ensemble_training", type=str2bool, default=False, help="Whether to use randomly sampled subsets to train classifiers of the ensemble"
)
exp_arg.add_argument(
    "--subset_ratio", type=float, default=1.0, help="The ratio of all samples of a batch randomly sampled as a subset to train classifiers of the ensemble"
)
exp_arg.add_argument(
    "--certainty_threshold", type=float, default=0, help="q-quantile [0,1] for calculating the threshold epsilon for filtering out samples"
)
exp_arg.add_argument(
    "--adaptive_certainty_threshold", type=str2bool, default=False, help="Whether to use failure rate to weight certatinty_threshold"
)
exp_arg.add_argument(
    "--ensemble_certainty_type", type=str, default="data_uncertainty", help="What type of uncertainty criterion for active learning; data_uncertainty or knowledge uncertainty."
)
exp_arg.add_argument(
    "--ensemble_certainty_threshold", type=float, default=1.0, help="q-quantile [0,1] for calculating the threshold epsilon for filtering out samples"
)
exp_arg.add_argument(
    "--apply_threshold_on_loss_classification", type=str2bool, default=True, help="Wether the threshold-based filtering is applied on calculating loss_classification"
)
exp_arg.add_argument(
    "--keep_batch_size_loss_ensemble", type=str2bool, default=False, help="Wether we keep the number of samples as the batch size for training classifiers when we use the threshold-based filtering"
)
exp_arg.add_argument(
    "--keep_batch_size_loss_classification", type=str2bool, default=False, help="Wether we keep the number of samples as the batch size for training classifiers when we use the threshold-based filtering"
)

# Glimpse network params
glimpse_arg = add_argument_group("Glimpse Network Params")
glimpse_arg.add_argument(
    "--patch_size", type=int, default=8, help="Size of extracted patch at highest res"
)
glimpse_arg.add_argument(
    "--glimpse_scale", type=int, default=1, help="Scale of successive patches"
)
glimpse_arg.add_argument(
    "--num_patches", type=int, default=1, help="# of downscaled patches per glimpse"
)
glimpse_arg.add_argument(
    "--loc_hidden", type=int, default=128, help="Hidden size of loc fc"
)
glimpse_arg.add_argument(
    "--glimpse_hidden", type=int, default=128, help="Hidden size of glimpse fc"
)


# Core network params
core_arg = add_argument_group("Core Network Params")
core_arg.add_argument(
    "--num_glimpses", type=int, default=6, help="# of glimpses, i.e. BPTT iterations"
)
core_arg.add_argument("--hidden_size", type=int, default=256, help="Hidden size of rnn")


# Reinforce params
reinforce_arg = add_argument_group("Reinforce Params")
reinforce_arg.add_argument(
    "--std", type=float, default=0.05, help="Gaussian policy standard deviation"
)
reinforce_arg.add_argument(
    "--M", type=int, default=1, help="Monte Carlo sampling for valid and test sets"
)
reinforce_arg.add_argument(
    "--reward_norm", type=str2bool, default=False, help="Whether to linearly normalize the reward in the range of [0,1]"
)

# Data params
data_arg = add_argument_group("Data Params")
data_arg.add_argument(
    "--valid_size",
    type=float,
    default=0.1,
    help="Proportion of training set used for validation",
)
data_arg.add_argument(
    "--batch_size", type=int, default=128, help="# of images in each batch of data"
)
data_arg.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="# of subprocesses to use for data loading",
)
data_arg.add_argument(
    "--shuffle",
    type=str2bool,
    default=True,
    help="Whether to shuffle the train and valid indices",
)
data_arg.add_argument(
    "--show_sample",
    type=str2bool,
    default=False,
    help="Whether to visualize a sample grid of the data",
)


# Training params
train_arg = add_argument_group("Training Params")
train_arg.add_argument(
    "--is_train", type=str2bool, default=True, help="Whether to train or test the model"
)
train_arg.add_argument(
    "--momentum", type=float, default=0.5, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default=1500, help="# of epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=3e-4, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=1500,
    help="Number of epochs to wait before reducing lr",
)
train_arg.add_argument(
    "--lr_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced. new_lr = lr * factor."
)
train_arg.add_argument(
    "--train_patience",
    type=int,
    default=1500,
    help="Number of epochs to wait before stopping train",
)


# Other params
misc_arg = add_argument_group("Misc.")
misc_arg.add_argument(
    "--use_gpu", type=str2bool, default=True, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--best",
    type=str2bool,
    default=True,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=1, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--exp_dir",
    type=str,
    default="./exps",
    help="Directory in which to save data for one experiment",
)
misc_arg.add_argument(
    "--data_dir", type=str, default="./data", help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt",
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default="./logs/",
    help="Directory in which Tensorboard logs wil be stored",
)
misc_arg.add_argument(
    "--use_tensorboard",
    type=str2bool,
    default=False,
    help="Whether to use tensorboard for visualization",
)
misc_arg.add_argument(
    "--resume_ckpt",
    type=str,
    default="",
    help="checkpoint to resume",
)
misc_arg.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help="How frequently to print training details",
)
misc_arg.add_argument(
    "--plot_freq", type=int, default=5, help="How frequently to plot glimpses"
)
misc_arg.add_argument(
    "--ckp_freq", type=int, default=0, help="How frequently to save checkpoints. 0 means we only save the best and final ckp. "
)
misc_arg.add_argument(
    "--save_init_ckpt", type=str2bool, default=True, help="Whether to save the randomly initialized checkpoint. "
)
misc_arg.add_argument(
    "--plot_training", type=str2bool, default=False, help="Whether to plot glimpses for training episodes"
)
misc_arg.add_argument(
    "--noise_visualization", type=str2bool, default=False, help="Whether to use the model with two classifiers for visualizing the noise. "
)
misc_arg.add_argument(
    "--oracle_ckpt",
    type=str,
    default="",
    help="Checkpoint of the oracle model to resume",
)
misc_arg.add_argument(
    "--reward_hacking",
    type=str2bool,
    default=False, 
    help="Whether to replace the reward by the reward from a pretained classifier to train the agent. ",
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

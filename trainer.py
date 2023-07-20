import os
import time
import shutil
import pickle

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.categorical import Categorical
from tensorboard_logger import configure, log_value

from model import RecurrentAttention, EarlyStoppingRecurrentAttention, RecurrentAttention_OracleClassifier
from utils import AverageMeter, set_requires_grad
import numpy as np

import wandb
from easydict import EasyDict as edict
from plot_glimpses import log_wandb_video
import itertools

class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_r_lat = config.weight_r_lat
        self.only_r_lat_on_success = config.only_r_lat_on_success
        self.use_acc_cert_weighted_r_lat = config.use_acc_cert_weighted_r_lat
        self.acc_reward_type = config.acc_reward_type
        self.use_implict_lat_penalty = config.use_implict_lat_penalty
        self.discount_factor_lat_penalty = config.discount_factor_lat_penalty
        self.use_information_maximization_reward = config.use_information_maximization_reward
        self.n_classifiers = config.n_classifiers
        self.subset_ensemble_training = config.subset_ensemble_training
        self.subset_ratio = config.subset_ratio
        self.certainty_threshold = config.certainty_threshold
        self.adaptive_certainty_threshold = config.adaptive_certainty_threshold
        self.apply_threshold_on_loss_classification = config.apply_threshold_on_loss_classification
        self.keep_batch_size_loss_ensemble = config.keep_batch_size_loss_ensemble
        self.keep_batch_size_loss_classification = config.keep_batch_size_loss_classification
        self.ensemble_certainty_type = config.ensemble_certainty_type
        self.ensemble_certainty_threshold = config.ensemble_certainty_threshold
        self.information_maximization_reward_type = config.information_maximization_reward_type
        self.data_uncertainty_based_resampling = config.data_uncertainty_based_resampling
        self.data_uncertainty_based_resampling_depth = config.data_uncertainty_based_resampling_depth
        self.random_baseline = config.random_baseline

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.lr_factor = config.lr_factor
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume_ckpt = config.resume_ckpt
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.ckp_freq = config.ckp_freq
        self.plot_training = config.plot_training
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )
        self.dynamic_exploration_time = config.dynamic_exploration_time
        self.use_wandb = config.use_wandb

        self.plot_dir = os.path.join(config.exp_dir, "./plots/" + self.model_name + "/")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        if self.dynamic_exploration_time: 
            self.model = EarlyStoppingRecurrentAttention(
                self.patch_size,
                self.num_patches,
                self.glimpse_scale,
                self.num_channels,
                self.loc_hidden,
                self.glimpse_hidden,
                self.std,
                self.hidden_size,
                self.num_classes,
                self.n_classifiers, 
            )
        else:
            if self.config.noise_visualization or self.config.reward_hacking:
                assert (self.config.noise_visualization and self.config.reward_hacking) != True
                self.model = RecurrentAttention_OracleClassifier(
                    self.patch_size,
                    self.num_patches,
                    self.glimpse_scale,
                    self.num_channels,
                    self.loc_hidden,
                    self.glimpse_hidden,
                    self.std,
                    self.hidden_size,
                    self.num_classes,
                    self.n_classifiers, 
                )
            else:
                self.model = RecurrentAttention(
                    self.patch_size,
                    self.num_patches,
                    self.glimpse_scale,
                    self.num_channels,
                    self.loc_hidden,
                    self.glimpse_hidden,
                    self.std,
                    self.hidden_size,
                    self.num_classes,
                    self.n_classifiers, 
                )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        if self.subset_ensemble_training:
            self.module_others = [self.model.sensor, self.model.rnn, self.model.locator, self.model.baseliner]
            self.optimizer_ensemble = torch.optim.Adam(
                self.model.ensemble_classifier.parameters(), lr=self.lr
            )
            self.optimizer_others = torch.optim.Adam(itertools.chain(*[m.parameters() for m in self.module_others]), lr=self.lr)
            self.scheduler_ensemble = ReduceLROnPlateau(
                self.optimizer_ensemble, "min", factor=self.lr_factor, patience=self.lr_patience
            )
            self.scheduler_others = ReduceLROnPlateau(
                self.optimizer_others, "min", factor=self.lr_factor, patience=self.lr_patience, min_lr=0.01 * self.lr
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, "min", factor=self.lr_factor, patience=self.lr_patience
            )

        # initialize wandb
        if self.use_wandb:
            wandb.init(project="noisy-rewards", config=vars(config))

    def reset(self, batch_size):
        h_t = torch.zeros(
            batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # save the randomly initialized checkpoint
        if self.config.save_init_ckpt:
            self.save_checkpoint(
                    {
                        "epoch": 1,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "best_valid_acc": self.best_valid_acc,
                    },
                    False,
                    0, 
                )

        # load checkpoint if needed
        if self.config.reward_hacking:
            if self.resume_ckpt != "":
                raise # have not been implemented
            else:
                self.load_checkpoint_reward_hacking()
        else:
            if self.resume_ckpt != "":
                self.load_checkpoint()
            else:
                pass

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )
        
        

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"] 
                        if not self.subset_ensemble_training else 
                        self.optimizer_others.param_groups[0]["lr"]
                )
            )

            # validate_logs = self.validate(epoch, "eval") # Debugging !!!

            # train for 1 epoch
            train_logs = self.train_one_epoch(epoch)

            # evaluate on validation set
            validate_logs = self.validate(epoch, "eval")
            # # reduce lr if validation loss plateaus
            if self.subset_ensemble_training:
                self.scheduler_ensemble.step(-validate_logs.accs.avg)
                self.scheduler_others.step(-validate_logs.accs.avg)
            else:
                self.scheduler.step(-validate_logs.accs.avg)

            is_best = validate_logs.accs.avg > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_logs.losses.avg, train_logs.accs.avg, validate_logs.losses.avg, validate_logs.accs.avg
                )
            )

            # save for wandb
            if self.use_wandb:
                header_wandb = ["train/epoch", "train/loss", "train/sl_loss", "train/rl_loss", "train/baseline_loss", 
                                "train/reward", "train/acc_reward", "train/lat_reward", "train/cert_reward", 
                                "train/acc", "train/steps", "train/grad_norm", "train/prob_gt", "train/var_ensemble", 
                                "train/lr", 
                                "eval/loss", "eval/sl_loss", "eval/rl_loss", "eval/baseline_loss", 
                                "eval/reward", "eval/acc_reward", "eval/lat_reward", 
                                "eval/acc", "eval/steps", "eval/prob_gt",
                                "eval/accs_uncertain", "eval/accs_resample", "eval/accs_all_origin", "eval/accs_all_resample", 
                                ]
                log_data = [epoch, train_logs.losses.avg, train_logs.sl_losses.avg, train_logs.rl_losses.avg, train_logs.baseline_losses.avg, 
                            train_logs.rewards.avg, train_logs.acc_rewards.avg, train_logs.lat_rewards.avg, train_logs.cert_rewards.avg, 
                            train_logs.accs.avg, train_logs.steps.avg,  train_logs.grad_norms.avg, train_logs.prob_gt.avg, train_logs.var_ensemble.avg, 
                            self.optimizer.param_groups[0]["lr"] if not self.subset_ensemble_training else self.optimizer_others.param_groups[0]["lr"], 
                            validate_logs.losses.avg, validate_logs.sl_losses.avg, validate_logs.rl_losses.avg, validate_logs.baseline_losses.avg, 
                            validate_logs.rewards.avg, validate_logs.acc_rewards.avg,  validate_logs.lat_rewards.avg, 
                            validate_logs.accs.avg, validate_logs.steps.avg, validate_logs.prob_gt.avg, 
                            validate_logs.accs_uncertain.avg, validate_logs.accs_resample.avg, validate_logs.accs_all_origin.avg, validate_logs.accs_all_resample.avg, 
                            ]
                wandb.log(dict(zip(header_wandb, log_data)))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(validate_logs.accs.avg, self.best_valid_acc)

            
            if self.ckp_freq > 0 and epoch != 0 and (epoch % self.ckp_freq) == 0 or epoch == (self.epochs - 1):
                if self.subset_ensemble_training:
                    self.save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "model_state": self.model.state_dict(),
                            "optim_ensemble_state": self.optimizer_ensemble.state_dict(),
                            "optim_others_state": self.optimizer_ensemble.state_dict(),
                            "best_valid_acc": self.best_valid_acc,
                        },
                        is_best,
                        epoch,
                    )
                else: 
                    self.save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "model_state": self.model.state_dict(),
                            "optim_state": self.optimizer.state_dict(),
                            "best_valid_acc": self.best_valid_acc,
                        },
                        is_best,
                        epoch, 
                    )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        logs = edict()
        logs.losses = AverageMeter()
        logs.sl_losses = AverageMeter()
        logs.rl_losses = AverageMeter()
        logs.baseline_losses = AverageMeter()
        logs.rewards = AverageMeter()
        logs.acc_rewards = AverageMeter()
        logs.lat_rewards = AverageMeter()
        logs.cert_rewards = AverageMeter()
        logs.accs = AverageMeter()
        logs.steps = AverageMeter()
        logs.grad_norms = AverageMeter()
        logs.prob_gt = AverageMeter()
        logs.var_ensemble = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                plot = False
                if self.plot_training and (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset(self.batch_size)
                if self.config.reward_hacking:
                    h_t_oracle = h_t

                # for plots
                imgs = []
                imgs.append(x[0:9])
                locs = []
                labels = []
                predictions = []

                # extract the glimpses
                l_log_pis = []
                baselines = []
                s_pis = []
                c_log_pis = []
                var_c_pis = []
                c_log_pis_ensemble = []

                # masks for trajectory tracking
                timestep_stop = torch.zeros(self.batch_size, dtype=torch.int, device=self.device, requires_grad=False)
                mask_done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device, requires_grad=False)

                if self.dynamic_exploration_time: 
                    # Dynamic exploration time
                    assert self.random_baseline == False 
                    all_stop_early = False
                    for t in range(self.num_glimpses):
                        h_t, l_t, l_log_pi, s_pi, b_t, c_log_pi = self.model(x, l_t, h_t)
                        # c_log_pi = torch.mean(c_log_pi_ensemble, 0) # mean of classifiers in the ensemble
                        c_log_pi_ensemble = c_log_pi.unsqueeze(0)
                        var_c_pi = torch.var(torch.exp(c_log_pi_ensemble), 0)
                        l_t_save = l_t.masked_fill(mask_done.unsqueeze(1).repeat(1,2), 10.0) # elements of l_t are in [-1,1]. 

                        stop = s_pi.detach() > 0.5
                        timestep_stop = timestep_stop + stop * t * (~mask_done)
                        mask_done = torch.logical_or(mask_done, stop)

                        # store
                        locs.append(l_t_save[0:9])
                        baselines.append(b_t)
                        l_log_pis.append(l_log_pi)
                        s_pis.append(s_pi)
                        c_log_pis.append(c_log_pi)
                        var_c_pis.append(var_c_pi)
                        c_log_pis_ensemble.append(c_log_pi_ensemble)

                        if torch.all(mask_done):
                            all_stop_early = True
                            break
                    
                    if not all_stop_early:
                        # give the final prediction if the agent dosen't choose to stop at `t=self.num_glimpses-1`
                        _, _, _, _, _, c_log_pi = self.model(x, l_t, h_t)
                        c_log_pi_ensemble = c_log_pi.unsqueeze(0)
                        # c_log_pi = torch.mean(c_log_pi_ensemble, 0) # mean of classifiers in the ensemble
                        var_c_pi = torch.var(torch.exp(c_log_pi_ensemble), 0)
                        t += 1; stop.fill_(1)
                        timestep_stop = timestep_stop + stop * t * (~mask_done)
                        mask_done = torch.logical_or(mask_done, stop)
                        c_log_pis.append(c_log_pi)
                        var_c_pis.append(var_c_pi)
                        c_log_pis_ensemble.append(c_log_pi_ensemble)

                    # actual steps
                    max_timestep_stop = torch.max(timestep_stop).item() # maximum equals to self.num_glimpses

                    # get mask_trajectory
                    mask_trajectory = torch.zeros(
                        self.batch_size,
                        max_timestep_stop + 1, 
                        dtype=torch.int,
                        device=self.device,
                        requires_grad=False,
                    ) # including the timestep of giving the stop action.
                    for b in range(self.batch_size):
                        mask_trajectory[b] = torch.tensor(
                                            [1] * (timestep_stop[b].item() + 1) + 
                                            [0] * (mask_trajectory.shape[1] - (timestep_stop[b].item() + 1)), 
                                            dtype=torch.int)

                    # convert list to tensors and reshape
                    baselines = torch.stack(baselines).transpose(1, 0)
                    l_log_pis = torch.stack(l_log_pis).transpose(1, 0)
                    s_pis = torch.stack(s_pis).transpose(1, 0)
                    c_log_pis = torch.stack(c_log_pis).transpose(1, 0)
                    var_c_pis = torch.stack(var_c_pis).transpose(1, 0)
                    c_log_pis_ensemble = torch.permute(torch.stack(c_log_pis_ensemble), (2,0,1,3))
                    
                    # merge l_log_pis and s_pis for policy learning
                    mask_stop = F.one_hot(timestep_stop, num_classes=max_timestep_stop+1).bool()
                    # s_log_pis_a = torch.nan_to_num(torch.log(s_pis), neginf=-1e+2)
                    s_log_pis_a = torch.log(s_pis)
                    s_log_pis = torch.mul(mask_stop[:,:s_log_pis_a.shape[1]], s_log_pis_a)
                    ls_log_pis_a = torch.mul(l_log_pis, ~mask_stop[:,:l_log_pis.shape[1]]) + s_log_pis
                    ls_log_pis = torch.mul(ls_log_pis_a, mask_trajectory[:,:ls_log_pis_a.shape[1]])

                    # calculate both information maximization-based reward and accuracy-based reward; we only use one of them as the real reward for RL training
                    c_log_pis = c_log_pis[mask_stop]
                    predicted = torch.max(c_log_pis, 1)[1]
                    var_c_pis = var_c_pis[mask_stop]
                    c_log_pi_ensemble = c_log_pis_ensemble[mask_stop]

                    # information maximization-based reward
                    y_onehot = F.one_hot(y, num_classes=10).bool()
                    c_log_pis_gt = c_log_pis[y_onehot]
                    c_pis_gt = torch.exp(c_log_pis_gt)

                    if self.information_maximization_reward_type == "linear_clipping": #"rectified_sub_plabel"
                        r_certainty = torch.clamp(c_pis_gt - 0.1, min=0, max=None)
                    elif self.information_maximization_reward_type == "linear_noclipping": # "chi_square": # "sub_plabel"
                        r_certainty = c_pis_gt - 0.1
                    elif self.information_maximization_reward_type == "log_clipping":
                        r_certainty = torch.clamp(torch.log(c_pis_gt) - np.log(0.1), min=0, max=None)
                    elif self.information_maximization_reward_type == "log_noclipping": # "log_plabel_over_prandom"
                        r_certainty = torch.log(c_pis_gt / 0.1)
                    else:
                        r_certainty = torch.zeros_like(c_pis_gt)

                    # latency reward
                    r_lat = (- timestep_stop / (self.num_glimpses + 1)).float()

                    # accuracy-based reward
                    if self.acc_reward_type == "minus_one_and_one": 
                        r_acc = ((predicted.detach() == y).float() - 0.5) * 2 # -1 for wrong prediction, 1 for correct prediction.
                    elif self.acc_reward_type == "zero_and_one": 
                        r_acc = (predicted.detach() == y).float()
                    else:
                        raise

                    # compute accuracy
                    correct = (predicted == y).float()
                    acc = 100 * (correct.sum() / len(y))

                    # calculte real reward for RL training and supervised learning loss
                    if self.use_information_maximization_reward: 
                        assert (self.certainty_threshold > 0) + (self.ensemble_certainty_threshold < 1) <= 1
                        filter_applied = False 
                        if self.certainty_threshold > 0: 
                            if self.adaptive_certainty_threshold:
                                # certainty_threshold = torch.quantile(r_certainty, (1 -  0.01 * acc) * self.certainty_threshold)
                                c_threshold = 0.01 * acc * self.certainty_threshold # max((acc - 0.1), 0) * self.certainty_threshold
                                certainty_threshold = torch.quantile(r_certainty, c_threshold)
                            else:
                                certainty_threshold = torch.quantile(r_certainty, self.certainty_threshold)
                            suf_mask = r_certainty > certainty_threshold
                            c_log_pis_filtered = c_log_pis[suf_mask]
                            c_log_pi_ensemble_filtered = c_log_pi_ensemble[suf_mask]
                            y_filtered = y[suf_mask]
                            filter_applied = True
                            
                        if self.apply_threshold_on_loss_classification and filter_applied: 
                            if self.keep_batch_size_loss_classification: 
                                n_samples = c_log_pis_filtered.shape[0]
                                indices = list(np.arange(n_samples)) + list(np.random.choice(n_samples, self.batch_size - n_samples, replace=True))
                                loss_classification = F.nll_loss(c_log_pis_filtered[indices], y_filtered[indices])
                            else: 
                                loss_classification = F.nll_loss(c_log_pis_filtered, y_filtered)
                        else:
                            loss_classification = F.nll_loss(c_log_pis, y)

                        if self.use_implict_lat_penalty:
                            R = r_certainty
                            R_discounted = torch.pow(self.discount_factor_lat_penalty, timestep_stop) * R
                            R_mtx = R_discounted.unsqueeze(1).repeat(1, baselines.shape[1])
                            R_mtx = torch.mul(R_mtx, mask_trajectory[:,:R_mtx.shape[1]])
                        else: 
                            if self.only_r_lat_on_success: 
                                R = r_certainty + self.weight_r_lat * torch.mul(r_certainty, r_lat) * (r_acc == 1.0)
                            elif self.use_acc_cert_weighted_r_lat: 
                                R = r_certainty + self.weight_r_lat * torch.mul(r_certainty, r_lat)
                            else:
                                R = r_certainty + self.weight_r_lat * r_lat
                            R_mtx = R.unsqueeze(1).repeat(1, baselines.shape[1])
                            R_mtx = torch.mul(R_mtx, mask_trajectory[:,:R_mtx.shape[1]])
                    else:
                        assert (self.only_r_lat_on_success and self.use_acc_cert_weighted_r_lat) == False
                        if self.use_implict_lat_penalty:
                            R = r_acc
                            R_discounted = torch.pow(self.discount_factor_lat_penalty, timestep_stop) * R
                            R_mtx = R_discounted.unsqueeze(1).repeat(1, baselines.shape[1])
                            R_mtx = torch.mul(R_mtx, mask_trajectory[:,:R_mtx.shape[1]])
                        else: 
                            if self.only_r_lat_on_success: 
                                R = r_acc + self.weight_r_lat * r_lat * (r_acc == 1.0)
                            elif self.use_acc_cert_weighted_r_lat:
                                correct = (predicted == y).float()
                                acc = correct.sum() / len(y)
                                R = r_acc + self.weight_r_lat * r_lat * acc
                            else:
                                R = r_acc + self.weight_r_lat * r_lat
                            R_mtx = R.unsqueeze(1).repeat(1, baselines.shape[1])
                            R_mtx = torch.mul(R_mtx, mask_trajectory[:,:R_mtx.shape[1]])

                        loss_classification = F.nll_loss(c_log_pis, y)

                    # mask baselines
                    baselines = torch.mul(baselines, mask_trajectory[:,:baselines.shape[1]])

                    # compute reinforce loss
                    adjusted_reward = R_mtx - baselines.detach()
                    loss_reinforce = torch.sum(-ls_log_pis * adjusted_reward, dim=1) # summed over timesteps and averaged across batch
                    loss_reinforce = torch.mean(loss_reinforce, dim=0)

                    # compute losses for the baseline module
                    loss_baseline = F.mse_loss(baselines, R_mtx, reduction="sum") / torch.sum(mask_trajectory)

                else: 
                    # Fixed exploration time
                    for t in range(self.num_glimpses - 1):
                        # forward pass through model
                        if self.config.reward_hacking:
                            h_t, l_t, b_t, p, h_t_oracle = self.model(x, l_t, h_t, h_t_oracle)
                        else:
                            h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                        # random baseline
                        if self.random_baseline:
                            l_t = torch.FloatTensor(l_t.shape[0], 2).uniform_(-1, 1).to(self.device)
                        # store
                        locs.append(l_t[0:9])
                        baselines.append(b_t)
                        l_log_pis.append(p)

                    # last iteration
                    if self.config.reward_hacking:
                        h_t, l_t, b_t, c_log_pi, c_log_pi_ensemble, p, c_log_pi_oracle = self.model(x, l_t, h_t, h_t_oracle, last=True)
                    else:
                        h_t, l_t, b_t, c_log_pi, c_log_pi_ensemble, p = self.model(x, l_t, h_t, last=True)
                    # c_log_pis = torch.mean(c_log_pi_ensemble, 0) # mean of classifiers in the ensemble
                    # c_log_pis = c_log_pi_ensemble[0]
                    # random baseline
                    if self.random_baseline:
                        l_t = torch.FloatTensor(l_t.shape[0], 2).uniform_(-1, 1).to(self.device)
                    if c_log_pi_ensemble == None:
                        c_log_pi_ensemble = c_log_pi.unsqueeze(0)
                    c_log_pis = c_log_pi
                    var_c_pis = torch.var(torch.exp(c_log_pi_ensemble), 0)
                    l_log_pis.append(p)
                    baselines.append(b_t)
                    locs.append(l_t[0:9])

                    # convert list to tensors and reshape
                    baselines = torch.stack(baselines).transpose(1, 0)
                    l_log_pis = torch.stack(l_log_pis).transpose(1, 0)
                    c_log_pi_ensemble = c_log_pi_ensemble.transpose(1, 0)

                    predicted = torch.max(c_log_pis, 1)[1]

                    # calculate both information maximization-based reward and accuracy-based reward; we only use one of them as the real reward for RL training
                    y_onehot = F.one_hot(y, num_classes=10).bool()
                    c_log_pis_gt = c_log_pis[y_onehot]
                    c_pis_gt = torch.exp(c_log_pis_gt)
                    if self.config.reward_hacking:
                        c_log_pis_gt_oracle = c_log_pi_oracle[y_onehot]
                        c_pis_gt_oracle = torch.exp(c_log_pis_gt_oracle)

                    if self.information_maximization_reward_type == "linear_clipping": #"rectified_sub_plabel"
                        r_certainty = torch.clamp(c_pis_gt - 0.1, min=0, max=None)
                        if self.config.reward_hacking:
                            r_certainty = torch.clamp(c_pis_gt_oracle - 0.1, min=0, max=None) 
                    elif self.information_maximization_reward_type == "linear_noclipping": # "chi_square": # "sub_plabel"
                        r_certainty = (c_pis_gt - 0.1) 
                        if self.config.reward_hacking:
                            r_certainty = (c_pis_gt_oracle - 0.1) 
                    elif self.information_maximization_reward_type == "log_clipping":
                        r_certainty = torch.clamp(torch.log(c_pis_gt) - np.log(0.1), min=0, max=None) 
                        if self.config.reward_hacking:
                            r_certainty = torch.clamp(torch.log(c_pis_gt_oracle) - np.log(0.1), min=0, max=None) 
                    elif self.information_maximization_reward_type == "log_noclipping": # "log_plabel_over_prandom"
                        r_certainty = torch.log(c_pis_gt / 0.1) 
                        if self.config.reward_hacking:
                            r_certainty = torch.log(c_pis_gt_oracle / 0.1) 
                    else:
                        raise
                        r_certainty = torch.zeros_like(c_pis_gt)
                    
                    # accuracy-based reward
                    if self.acc_reward_type == "minus_one_and_one": 
                        r_acc = ((predicted.detach() == y).float() - 0.5) * 2 # -1 for wrong prediction, 1 for correct prediction.
                    elif self.acc_reward_type == "zero_and_one": 
                        r_acc = (predicted.detach() == y).float()
                        if self.config.reward_hacking:
                            predicted_oracle = torch.max(c_log_pi_oracle, 1)[1]
                            r_acc = (predicted_oracle.detach() == y).float()
                    else:
                        raise

                    # compute accuracy
                    correct = (predicted == y).float()
                    acc = 100 * (correct.sum() / len(y))
                    
                    # calculte real reward for RL training and supervised learning loss
                    if self.use_information_maximization_reward:
                        # mask for filtering out insufficient samples for calculating the classification loss
                        filter_applied = False 
                        assert (self.certainty_threshold > 0) + (self.ensemble_certainty_threshold < 1) <= 1
                        if self.certainty_threshold > 0: 
                            if self.adaptive_certainty_threshold:
                                # certainty_threshold = torch.quantile(r_certainty, (1 -  0.01 * acc) * self.certainty_threshold)
                                c_threshold = 0.01 * acc * self.certainty_threshold # max((acc - 0.1), 0) * self.certainty_threshold
                                certainty_threshold = torch.quantile(r_certainty, c_threshold)
                            else:
                                certainty_threshold = torch.quantile(r_certainty, self.certainty_threshold)
                            suf_mask = r_certainty > certainty_threshold
                            c_log_pis_filtered = c_log_pis[suf_mask]
                            c_log_pi_ensemble_filtered = c_log_pi_ensemble[suf_mask]
                            y_filtered = y[suf_mask]
                            filter_applied = True
                        
                        elif self.ensemble_certainty_threshold < 1:
                            # total_uncertainty: H_E_P
                            # expected_data_uncertainty: E_H_P
                            # c_log_pis_mean = torch.mean(c_log_pi_ensemble, 1) # mean of classifiers in the ensemble
                            # total_uncertainty = Categorical(c_log_pis_mean).entropy().detach() # .mean()
                            assert self.n_classifiers > 1
                            expected_data_uncertainty = []
                            for i_classifier in range(self.n_classifiers - 1): 
                                c_log_pi_i_classifier = c_log_pi_ensemble[:,i_classifier,:]
                                expected_data_uncertainty.append(Categorical(c_log_pi_i_classifier).entropy().detach()) # .mean()
                            expected_data_uncertainty = torch.stack(expected_data_uncertainty).mean(0)
                            # uncertainty_score = total_uncertainty - expected_data_uncertainty # [0,1], 0 means predictions from multiple classifiers are very consistant/certain. Could it be negative?
                            if self.ensemble_certainty_type == "data_uncertainty":
                                if self.adaptive_certainty_threshold:
                                    c_threshold = 1 - (1 - self.ensemble_certainty_threshold) * (0.01 * acc)
                                    uncertainty_threshold = torch.quantile(expected_data_uncertainty, c_threshold)
                                else: 
                                    # certainty_threshold = torch.quantile(uncertainty_score, self.ensemble_certainty_threshold)
                                    uncertainty_threshold = torch.quantile(expected_data_uncertainty, self.ensemble_certainty_threshold) # debug
                                # suf_mask = uncertainty_score < certainty_threshold
                                suf_mask = expected_data_uncertainty < uncertainty_threshold # debug
                            else:
                                raise
                            c_log_pis_filtered = c_log_pis[suf_mask]
                            c_log_pi_ensemble_filtered = c_log_pi_ensemble[suf_mask]
                            y_filtered = y[suf_mask]
                            filter_applied = True

                        if self.apply_threshold_on_loss_classification and filter_applied: 
                            if self.keep_batch_size_loss_classification: 
                                n_samples = c_log_pis_filtered.shape[0]
                                indices = list(np.arange(n_samples)) + list(np.random.choice(n_samples, self.batch_size - n_samples, replace=True))
                                loss_classification = F.nll_loss(c_log_pis_filtered[indices], y_filtered[indices])
                            else: 
                                loss_classification = F.nll_loss(c_log_pis_filtered, y_filtered)
                        else:
                            loss_classification = F.nll_loss(c_log_pis, y)

                        R = r_certainty
                    else:
                        R = r_acc
                        loss_classification = F.nll_loss(c_log_pis, y)
                    
                    R_mtx = R.unsqueeze(1).repeat(1, self.num_glimpses)

                    # for logging
                    r_lat = torch.zeros(R_mtx.shape, dtype=torch.float, device=self.device)
                    timestep_stop.fill_(self.num_glimpses - 1)

                    # compute losses for differentiable modules
                    loss_baseline = F.mse_loss(baselines, R_mtx)

                    # compute reinforce loss
                    # summed over timesteps and averaged across batch
                    adjusted_reward = R_mtx - baselines.detach()
                    loss_reinforce = torch.sum(-l_log_pis * adjusted_reward, dim=1)
                    loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # gradient-based optimization
                if self.subset_ensemble_training:
                    # calculate gradients of ensemble
                    loss_ensemble = []
                    set_requires_grad(self.module_others, False)
                    set_requires_grad(self.model.ensemble_classifier, True)
                    self.optimizer_ensemble.zero_grad()

                    n_samples = c_log_pis_filtered.shape[0] if filter_applied else c_log_pis.shape[0]
                    for i_classifier in range(self.n_classifiers):
                        if self.keep_batch_size_loss_ensemble:
                            # indices = np.random.choice(n_samples, self.batch_size, replace=True)
                            indices = list(np.arange(n_samples)) + list(np.random.choice(n_samples, self.batch_size - n_samples, replace=True))
                        else:
                            indices = list(np.arange(n_samples))
                            # indices = np.random.choice(n_samples, int(self.subset_ratio * n_samples), replace=True)
                        if filter_applied: 
                            loss_ensemble_i = F.nll_loss(c_log_pi_ensemble_filtered[indices,i_classifier,:], y_filtered[indices])
                        else:
                            loss_ensemble_i = F.nll_loss(c_log_pi_ensemble[indices,i_classifier,:], y[indices])
                        loss_ensemble.append(loss_ensemble_i)
                    loss_ensemble = sum(loss_ensemble)
                    loss_ensemble.backward(retain_graph = True)
                    
                    # calculate gradients of other modules; sum up into a hybrid loss
                    loss = loss_classification + loss_baseline + loss_reinforce * 0.01
                    set_requires_grad(self.module_others, True)
                    set_requires_grad(self.model.ensemble_classifier, False)
                    self.optimizer_others.zero_grad()
                    loss.backward()
                    grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5

                    # update parameters
                    self.optimizer_ensemble.step()
                    self.optimizer_others.step()
                else:
                    # sum up into a hybrid loss
                    # Debug: subset_ensemble_training False for ensemble
                    if self.n_classifiers > 1: 
                        loss_classification_ensemble = []
                        n_samples = c_log_pis_filtered.shape[0] if filter_applied else c_log_pis.shape[0]

                        # if filter_applied: 
                        #     loss_classification_classifier = F.nll_loss(c_log_pis_filtered[indices,:], y_filtered[indices])
                        # else:
                        #     loss_classification_classifier = F.nll_loss(c_log_pis[indices,:], y[indices])

                        for i_ensemble_classifier in range(self.n_classifiers - 1):
                            # indices = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
                            indices = np.random.choice(n_samples, n_samples, replace=True)
                            if filter_applied: 
                                loss_classification_ensemble_i = F.nll_loss(c_log_pi_ensemble_filtered[indices,i_ensemble_classifier,:], y_filtered[indices])
                            else:
                                loss_classification_ensemble_i = F.nll_loss(c_log_pi_ensemble[indices,i_ensemble_classifier,:], y[indices])
                            loss_classification_ensemble.append(loss_classification_ensemble_i)
                        loss_classification_ensemble = sum(loss_classification_ensemble)
                        loss = loss_classification + loss_classification_ensemble + loss_baseline + loss_reinforce * 0.01
                    else: 
                        # loss_classification has been calcultaed before
                        loss = loss_classification + loss_baseline + loss_reinforce * 0.01
                        pass

                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
                    self.optimizer.step()

                # for plots
                labels.append(y[0:9])
                predictions.append(predicted[0:9])

                # update logs
                logs.losses.update(loss.item(), x.size()[0])
                logs.sl_losses.update(loss_classification.item(), x.size()[0])
                logs.rl_losses.update(loss_reinforce.item(), x.size()[0])
                logs.baseline_losses.update(loss_baseline.item(), x.size()[0])
                logs.rewards.update(R.mean().item(), x.size()[0])
                logs.acc_rewards.update(r_acc.mean().item(), x.size()[0])
                logs.lat_rewards.update(r_lat.mean().item(), x.size()[0])
                logs.cert_rewards.update(r_certainty.mean().item(), x.size()[0])
                logs.accs.update(acc.item(), x.size()[0])
                logs.steps.update(timestep_stop.float().mean().item(), x.size()[0])
                logs.grad_norms.update(grad_norm, x.size()[0])
                logs.prob_gt.update(c_pis_gt.mean().item(), x.size()[0])
                logs.var_ensemble.update(var_c_pis.mean().item(), x.size()[0])

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                # update pbar
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f} - steps: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item(), timestep_stop.float().mean().item(),
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    labels = [y.cpu().data.numpy() for y in labels]
                    predictions = [p.cpu().data.numpy() for p in predictions]
                    pickle.dump(
                        imgs, open(self.plot_dir + "train_g_{}.p".format(epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "train_l_{}.p".format(epoch + 1), "wb")
                    )
                    # log video in wandb
                    if self.use_wandb:
                        log_wandb_video(imgs, locs, labels, predictions, self.patch_size, "train")

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", logs.losses.avg, iteration)
                    log_value("train_acc", logs.accs.avg, iteration)
                    log_value("train_steps", logs.steps.avg, iteration)

            return logs

    @torch.no_grad()
    def validate(self, epoch, eval_or_test):
        """Evaluate the RAM model on the validation set.
        """
        self.model.eval()

        batch_time = AverageMeter()
        logs = edict()
        logs.losses = AverageMeter()
        logs.sl_losses = AverageMeter()
        logs.rl_losses = AverageMeter()
        logs.baseline_losses = AverageMeter()
        logs.rewards = AverageMeter()
        logs.acc_rewards = AverageMeter()
        logs.lat_rewards = AverageMeter()
        logs.accs = AverageMeter()
        logs.steps = AverageMeter()
        logs.prob_gt = AverageMeter()

        # for debugging
        logs.accs_uncertain = AverageMeter()
        logs.accs_resample = AverageMeter()
        logs.accs_all_origin = AverageMeter()
        logs.accs_all_resample = AverageMeter()

        tic = time.time()
        assert eval_or_test in ["eval", "test"]

        # for visualizing noise
        if self.config.noise_visualization:
            assert eval_or_test == "test"
            logs.noise_list = []
            logs.accs_oracle = AverageMeter()

        if eval_or_test == "eval":
            dataloader = self.valid_loader
            num_samples = self.num_valid
        else:
            dataloader = self.test_loader
            num_samples = self.num_test

        with tqdm(total=num_samples) as pbar:
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)

                plot = False
                if eval_or_test == "eval": 
                    if (epoch % self.plot_freq == 0) and (i == 0):
                        plot = True
                else:
                    if i == 0: 
                        plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset(self.batch_size)
                if self.config.noise_visualization or self.config.reward_hacking:
                    h_t_oracle = h_t

                # for plots
                imgs = []
                imgs.append(x[0:9])
                locs = []
                labels = []
                predictions = []

                # extract the glimpses
                l_log_pis = []
                baselines = []
                s_pis = []
                c_log_pis = []

                # masks for trajectory tracking
                mask_trajectory = torch.zeros(
                    self.batch_size,
                    self.num_glimpses,
                    dtype=torch.int,
                    device=self.device,
                    requires_grad=False,
                ) # including the timestep of giving the stop action.
                timestep_stop = torch.zeros(self.batch_size, dtype=torch.int, device=self.device, requires_grad=False)
                mask_done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device, requires_grad=False)

                if self.dynamic_exploration_time: 
                    # Early stopping
                    for t in range(self.num_glimpses):
                        h_t, l_t, l_log_pi, s_pi, b_t, c_log_pi = self.model(x, l_t, h_t)
                        c_log_pi_ensemble = c_log_pi.unsqueeze(0)
                        l_t_save = l_t.masked_fill(mask_done.unsqueeze(1).repeat(1,2), 10.0) # elements of l_t are in [-1,1]. 

                        stop = s_pi.detach() > 0.5
                        timestep_stop = timestep_stop + stop * t * (~mask_done)
                        mask_done = torch.logical_or(mask_done, stop)

                        # store
                        locs.append(l_t_save[0:9])
                        baselines.append(b_t)
                        l_log_pis.append(l_log_pi)
                        s_pis.append(s_pi)
                        c_log_pis.append(c_log_pi)

                    # give the final prediction if the agent dosen't choose to stop at `t=self.num_glimpses-1`
                    _, _, _, _, _, c_log_pi = self.model(x, l_t, h_t)
                    c_log_pi_ensemble = c_log_pi.unsqueeze(0)
                    t += 1; stop.fill_(1)
                    timestep_stop = timestep_stop + stop * t * (~mask_done)
                    mask_done = torch.logical_or(mask_done, stop)
                    c_log_pis.append(c_log_pi)

                    # get mask_trajectory
                    for b in range(self.batch_size):
                        mask_trajectory[b] = torch.tensor([1] * min(timestep_stop[b].item() + 1, self.num_glimpses) + 
                                            [0] * (self.num_glimpses - min(timestep_stop[b].item() + 1, self.num_glimpses)), 
                                            dtype=torch.int)

                    # convert list to tensors and reshape
                    baselines = torch.stack(baselines).transpose(1, 0)
                    l_log_pis = torch.stack(l_log_pis).transpose(1, 0)
                    s_pis = torch.stack(s_pis).transpose(1, 0)
                    c_log_pis = torch.stack(c_log_pis).transpose(1, 0)
                    
                    # merge l_log_pis and s_pis for policy learning
                    mask_stop = F.one_hot(timestep_stop, num_classes=self.num_glimpses+1).bool()
                    # s_log_pis_a = torch.nan_to_num(torch.log(s_pis), neginf=-1e+2)
                    s_log_pis_a = torch.log(s_pis)
                    s_log_pis = torch.mul(mask_stop[:,:self.num_glimpses], s_log_pis_a)
                    ls_log_pis_a = torch.mul(l_log_pis, ~mask_stop[:,:self.num_glimpses]) + s_log_pis
                    ls_log_pis = torch.mul(ls_log_pis_a, mask_trajectory)

                    # calculate reward
                    c_log_pis = c_log_pis[mask_stop]
                    predicted = torch.max(c_log_pis, 1)[1]
                    # accuracy-based reward
                    if self.acc_reward_type == "minus_one_and_one": 
                        r_acc = ((predicted.detach() == y).float() - 0.5) * 2 # -1 for wrong prediction, 1 for correct prediction.
                    elif self.acc_reward_type == "zero_and_one": 
                        r_acc = (predicted.detach() == y).float()
                    else:
                        raise
                    r_lat = (- timestep_stop / (self.num_glimpses + 1)).float()
                    R = r_acc + self.weight_r_lat * r_lat
                    R_mtx = R.unsqueeze(1).repeat(1, self.num_glimpses)
                    R_mtx = torch.mul(R_mtx, mask_trajectory)

                    # mask baselines
                    baselines = torch.mul(baselines, mask_trajectory)

                    # compute losses for differentiable modules
                    loss_classification = F.nll_loss(c_log_pis, y)
                    loss_baseline = F.mse_loss(baselines, R_mtx)

                    # compute reinforce loss
                    # summed over timesteps and averaged across batch
                    adjusted_reward = R_mtx - baselines.detach()
                    loss_reinforce = torch.sum(-ls_log_pis * adjusted_reward, dim=1)
                    loss_reinforce = torch.mean(loss_reinforce, dim=0)

                    # compute accuracy
                    predicted = torch.max(c_log_pis, 1)[1]
                    correct = (predicted == y).float()
                    acc = 100 * (correct.sum() / len(y))

                else: 
                    # No early stopping
                    for t in range(self.num_glimpses - 1):
                        # forward pass through model
                        if self.config.noise_visualization or self.config.reward_hacking:
                            h_t, l_t, b_t, p, h_t_oracle = self.model(x, l_t, h_t, h_t_oracle)
                        else:
                            h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                        # random baseline
                        if self.random_baseline:
                            l_t = torch.FloatTensor(l_t.shape[0], 2).uniform_(-1, 1).to(self.device)
                        # store
                        locs.append(l_t[0:9])
                        baselines.append(b_t)
                        l_log_pis.append(p)

                    # last iteration
                    if self.config.noise_visualization or self.config.reward_hacking:
                        h_t, l_t, b_t, c_log_pi, c_log_pi_ensemble, p, c_log_pi_oracle = self.model(x, l_t, h_t, h_t_oracle, last=True)
                    else:
                        h_t, l_t, b_t, c_log_pi, c_log_pi_ensemble, p = self.model(x, l_t, h_t, last=True)
                    # random baseline
                    if self.random_baseline:
                        l_t = torch.FloatTensor(l_t.shape[0], 2).uniform_(-1, 1).to(self.device)
                    # c_log_pis = torch.mean(c_log_pi_ensemble, 0) # mean of classifiers in the ensemble
                    if c_log_pi_ensemble == None:
                        c_log_pi_ensemble = c_log_pi.unsqueeze(0)
                    c_log_pis = c_log_pi
                    l_log_pis.append(p)
                    baselines.append(b_t)
                    locs.append(l_t[0:9])

                    c_log_pi_ensemble = c_log_pi_ensemble.transpose(1, 0)

                    # data_uncertainty_based_resampling = True # Debug !!!
                    if self.data_uncertainty_based_resampling and self.ensemble_certainty_threshold < 1: 
                        # c_log_pis_mean = torch.mean(c_log_pi_ensemble, 1) # mean of classifiers in the ensemble
                        # total_uncertainty = Categorical(c_log_pis_mean).entropy().detach() # .mean()
                        assert self.n_classifiers > 1
                        expected_data_uncertainty = []
                        for i_classifier in range(self.n_classifiers - 1): 
                            c_log_pi_i_classifier = c_log_pi_ensemble[:,i_classifier,:]
                            expected_data_uncertainty.append(Categorical(c_log_pi_i_classifier).entropy().detach()) # .mean()
                        expected_data_uncertainty = torch.stack(expected_data_uncertainty).mean(0)
                        # uncertainty_score = total_uncertainty - expected_data_uncertainty # [0,1], 0 means predictions from multiple classifiers are very consistant/certain. Could it be negative?
                        if self.ensemble_certainty_type == "data_uncertainty":
                            if self.adaptive_certainty_threshold:
                                c_threshold = 1 - (1 - self.ensemble_certainty_threshold) * (0.01 * acc)
                                uncertainty_threshold = torch.quantile(expected_data_uncertainty, c_threshold)
                            else: 
                                # certainty_threshold = torch.quantile(uncertainty_score, self.ensemble_certainty_threshold)
                                uncertainty_threshold = torch.quantile(expected_data_uncertainty, self.ensemble_certainty_threshold)
                            # suf_mask = uncertainty_score < certainty_threshold
                            suf_mask = expected_data_uncertainty < uncertainty_threshold
                        else:
                            raise
                        
                        c_log_pis_filtered = c_log_pis[suf_mask]
                        c_log_pi_ensemble_filtered = c_log_pi_ensemble[suf_mask]
                        y_filtered = y[suf_mask]
                        filter_applied = True

                        # calculate accuracy
                        predicted_filtered = torch.max(c_log_pis_filtered, 1)[1]
                        correct_filtered = (predicted_filtered == y_filtered).float()
                        # acc_filtered = 100 * (correct_filtered.sum() / len(y_filtered))

                        # For debugging
                        c_log_pis_uncertain = c_log_pis[~suf_mask]
                        y_uncertain = y[~suf_mask]
                        predicted_uncertain = torch.max(c_log_pis_uncertain, 1)[1]
                        correct_uncertain = (predicted_uncertain == y_uncertain).float()
                        acc_uncertain = correct_uncertain.sum() / sum(~suf_mask)
                        correct_all = (torch.max(c_log_pis, 1)[1] == y).float()
                        acc_all = correct_all.sum() / y.shape[0] # self.batch_size
                        # uncertainty_score_sorted, uncertainty_score_sorted_index = uncertainty_score.sort()
                        # correct_all_sorted = correct_all[uncertainty_score_sorted_index]

                        # resample trajectories
                        num_resample_total = sum(~suf_mask) 
                        correct_resample_total_num = 0
                        num_resample = num_resample_total
                        l_t_pre = l_t
                        x_pre = x
                        y_pre = y
                        for resample_i in range(self.data_uncertainty_based_resampling_depth):
                            h_t = torch.zeros(
                                num_resample, # not self.batch_size
                                self.hidden_size,
                                dtype=torch.float,
                                device=self.device,
                                requires_grad=True,
                            )
                            l_t = l_t_pre[~suf_mask]
                            # l_t_pre = l_t
                            # h_t, l_t = self.reset(num_resample)
                            
                            # l_t.requires_grad = True
                            x_resample = x_pre[~suf_mask]
                            y_resample = y_pre[~suf_mask]
                            for t in range(self.num_glimpses - 1):
                                h_t, l_t, b_t, p = self.model(x_resample, l_t, h_t)
                                # locs.append(l_t[0:9])
                                # baselines.append(b_t)
                                # l_log_pis.append(p)
                            h_t, l_t, b_t, c_log_pis_resample, c_log_pi_ensemble_resample, p = self.model(x_resample, l_t, h_t, last=True)
                            if c_log_pi_ensemble_resample == None:
                                c_log_pi_ensemble_resample = c_log_pi.unsqueeze(0)
                            c_log_pi_ensemble_resample = c_log_pi_ensemble_resample.transpose(1, 0)
                            
                            # re-compuate data uncertainty
                            expected_data_uncertainty = []
                            for i_classifier in range(self.n_classifiers - 1): 
                                c_log_pi_i_classifier = c_log_pi_ensemble_resample[:,i_classifier,:]
                                expected_data_uncertainty.append(Categorical(c_log_pi_i_classifier).entropy().detach()) # .mean()
                            expected_data_uncertainty = torch.stack(expected_data_uncertainty).mean(0)

                            # judge correctness for samples with low uncertainty
                            suf_mask = expected_data_uncertainty < uncertainty_threshold # we use the original threshold calculated from the batch of samples
                            predicted_resample = torch.max(c_log_pis_resample, 1)[1]
                            if resample_i == self.data_uncertainty_based_resampling_depth - 1: 
                                correct_resample = (predicted_resample == y_resample).float()
                            else:
                                correct_resample = (predicted_resample[suf_mask] == y_resample[suf_mask]).float()
                            correct_resample_total_num += correct_resample.sum()

                            # update variables for recursion
                            num_resample = sum(~suf_mask) 
                            l_t_pre = l_t
                            x_pre = x_resample
                            y_pre = y_resample
                            if num_resample == 0:
                                break
                            
                        # for debugging
                        acc_resample = correct_resample_total_num / num_resample_total

                        # calculate final accuracy
                        # acc = torch.cat((correct_filtered, correct_resample_total_num), 0).sum() / self.batch_size
                        acc = (correct_filtered.sum() + correct_resample_total_num) / y.shape[0] # self.batch_size

                        # for debugging
                        logs.accs_uncertain.update(acc_uncertain.item(), y.shape[0])
                        logs.accs_resample.update(acc_resample.item(), y.shape[0])
                        logs.accs_all_origin.update(acc_all.item(), y.shape[0])
                        logs.accs_all_resample.update(acc.item(), y.shape[0])

                    else:
                        # compute accuracy
                        predicted = torch.max(c_log_pis, 1)[1]
                        correct = (predicted == y).float()
                        acc = 100 * (correct.sum() / len(y))
                        if self.config.noise_visualization:
                            predicted_oracle = torch.max(c_log_pi_oracle, 1)[1]
                            correct_oracle = (predicted_oracle == y).float()
                            acc_oracle = 100 * (correct_oracle.sum() / len(y))

                    # convert list to tensors and reshape
                    baselines = torch.stack(baselines).transpose(1, 0)
                    l_log_pis = torch.stack(l_log_pis).transpose(1, 0)

                    # calculate reward
                    predicted = torch.max(c_log_pis, 1)[1]
                    # accuracy-based reward
                    if self.acc_reward_type == "minus_one_and_one": 
                        r_acc = ((predicted.detach() == y).float() - 0.5) * 2 # -1 for wrong prediction, 1 for correct prediction.
                    elif self.acc_reward_type == "zero_and_one": 
                        r_acc = (predicted.detach() == y).float()
                    else:
                        raise
                    R = r_acc
                    R = R.unsqueeze(1).repeat(1, self.num_glimpses)

                    # for logging
                    r_lat = torch.zeros(R.shape, dtype=torch.float, device=self.device)
                    timestep_stop.fill_(self.num_glimpses - 1)

                    # compute losses for differentiable modules
                    loss_classification = F.nll_loss(c_log_pis, y)
                    loss_baseline = F.mse_loss(baselines, R)

                    # compute reinforce loss
                    # summed over timesteps and averaged across batch
                    adjusted_reward = R - baselines.detach()
                    loss_reinforce = torch.sum(-l_log_pis * adjusted_reward, dim=1)
                    loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_classification + loss_baseline + loss_reinforce * 0.01

                
                # for plots
                labels.append(y[0:9])
                predictions.append(predicted[0:9])

                # log the probability of the ground truth label
                y_onehot = F.one_hot(y, num_classes=10).bool()
                c_log_pis_gt = c_log_pis[y_onehot]
                c_pis_gt = torch.exp(c_log_pis_gt)

                if self.config.noise_visualization:
                    c_log_pis_gt_oracle = c_log_pi_oracle[y_onehot]
                    c_pis_gt_oracle = torch.exp(c_log_pis_gt_oracle)

                # update logs
                logs.losses.update(loss.item(), x.size()[0])
                logs.sl_losses.update(loss_classification.item(), x.size()[0])
                logs.rl_losses.update(loss_reinforce.item(), x.size()[0])
                logs.baseline_losses.update(loss_baseline.item(), x.size()[0])
                logs.rewards.update(R.mean().item(), x.size()[0])
                logs.acc_rewards.update(r_acc.mean().item(), x.size()[0])
                logs.lat_rewards.update(r_lat.mean().item(), x.size()[0])
                logs.accs.update(acc.item(), x.size()[0])
                logs.steps.update(timestep_stop.float().mean().item(), x.size()[0])
                logs.prob_gt.update(c_pis_gt.mean().item(), x.size()[0])

                if self.config.noise_visualization:
                    # logs.prob_gt_list.append(c_pis_gt.detach().cpu().numpy())
                    noise = c_pis_gt - c_pis_gt_oracle 
                    logs.noise_list.append(noise.detach().cpu().numpy())
                    logs.accs_oracle.update(acc_oracle.mean().item(), x.size()[0])

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                # update pbar
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f} - steps: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item(), timestep_stop.float().mean().item(),
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    labels = [y.cpu().data.numpy() for y in labels]
                    predictions = [p.cpu().data.numpy() for p in predictions]
                    pickle.dump(
                        imgs, open(self.plot_dir + "{}_g_{}.p".format(eval_or_test, epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "{}_l_{}.p".format(eval_or_test, epoch + 1), "wb")
                    )
                    # log video in wandb
                    if self.use_wandb:
                        log_wandb_video(imgs, locs, labels, predictions, self.patch_size, "eval")

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("valid_loss", logs.losses.avg, iteration)
                    log_value("valid_loss", logs.accs.avg, iteration)
                    log_value("valid_steps", logs.steps.avg, iteration)
            
            return logs

    @torch.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        # correct = 0

        # load the best checkpoint
        assert self.resume_ckpt != ""
        if self.config.noise_visualization:
            self.load_checkpoint_noise_visualization()
        else:
            self.load_checkpoint()

        print(
            "\n[*] Test on {} samples".format(
                self.num_test
            )
        )

        logs = self.validate(epoch=0, eval_or_test="test")
      
        print("accs: ", logs.accs.avg)
        if self.config.noise_visualization:
            print("accs_oracle: ", logs.accs_oracle.avg)
            # for visualizing the noise 
            # prob_gt_array = np.concatenate(logs.prob_gt_list, axis=0)
            # np.save(os.path.join(self.logs_dir, "prob_gt_array_{}.npy".format(self.start_epoch - 1)), prob_gt_array)
            noise_array = np.concatenate(logs.noise_list, axis=0)
            np.save(os.path.join(self.logs_dir, "noise_array_{}.npy".format(self.start_epoch - 1)), noise_array)


    def save_checkpoint(self, state, is_best, epoch=None):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt_{}.pth.tar".format(epoch)
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        # filename = self.resume_ckpt
        # ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(self.resume_ckpt)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        # if self.subset_ensemble_training: # Debug !!!
        #     self.optimizer_ensemble.load_state_dict(ckpt["optim_ensemble_state"])
        #     self.optimizer_others.load_state_dict(ckpt["optim_others_state"])
        # else: 
        #     self.optimizer.load_state_dict(ckpt["optim_state"])

        print("[*] Loaded {} checkpoint @ epoch {}".format(self.resume_ckpt, ckpt["epoch"]))

    def load_checkpoint_noise_visualization(self):
        """load the classifier of a converged model as the oracle classfier.
        load the classifier of a arbitrary model to visualize the difference/noise between these two classifiers. 
        """
        assert self.config.dynamic_exploration_time == False # we only implement the noise visualization code fot models with fixed movement steps
        oracle_model = torch.load(self.config.oracle_ckpt)
        target_model = torch.load(self.resume_ckpt)

        # construct a state dict for the RecurrentAttention_NyiwoiseVisualization model
        state_oracle = oracle_model["model_state"]
        state_target = target_model["model_state"]

        for k in self.model.state_dict().keys():
            if "oracle" in k:
                state_target[k] = state_oracle[k.replace("_oracle", "")]

        # state_target["classifier_oracle.fc.weight"] = state_oracle["classifier.fc.weight"]
        # state_target["classifier_oracle.fc.bias"] = state_oracle["classifier.fc.bias"]

        # load the constructed parameters into the model
        self.model.load_state_dict(state_target)

        print("[*] Loading the oracle model from {}".format(self.config.oracle_ckpt))
        print("[*] Loading the target model from {}".format(self.resume_ckpt))

        # filename = self.resume_ckpt
        # ckpt_path = os.path.join(self.ckpt_dir, filename)

        # load variables from checkpoint
        self.start_epoch = target_model["epoch"]
        self.best_valid_acc = target_model["best_valid_acc"]

        print("[*] Loaded oracle {} checkpoint @ epoch {}".format(self.config.oracle_ckpt, oracle_model["epoch"]))
        print("[*] Loaded target {} checkpoint @ epoch {}".format(self.resume_ckpt, target_model["epoch"]))

    def load_checkpoint_reward_hacking(self):
        """load the classifier of a converged model as the oracle classfier.
        Keep other modules randomly initialized. 
        """
        assert self.config.dynamic_exploration_time == False # we only implement the noise visualization code fot models with fixed movement steps
        oracle_model = torch.load(self.config.oracle_ckpt)

        # construct a state dict for the RecurrentAttention_NyiwoiseVisualization model
        state_oracle = oracle_model["model_state"]
        state_model = self.model.state_dict()
        for k in self.model.state_dict().keys():
            if "oracle" in k:
                state_model[k] = state_oracle[k.replace("_oracle", "")]

        # state_target["classifier_oracle.fc.weight"] = state_oracle["classifier.fc.weight"]
        # state_target["classifier_oracle.fc.bias"] = state_oracle["classifier.fc.bias"]

        # load the constructed parameters into the model
        print("[*] Loading the oracle model from {}".format(self.config.oracle_ckpt))

        self.model.load_state_dict(state_model)

        print("[*] Loaded oracle {} checkpoint @ epoch {}".format(self.config.oracle_ckpt, oracle_model["epoch"]))

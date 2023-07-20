import torch
import torch.nn as nn

import modules

class RecurrentAttention_OracleClassifier(nn.Module):
    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes, n_classifiers, 
    ):
        super().__init__()

        self.std = std

        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = modules.CoreNetwork(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)
        self.classifier = modules.ClassificationNetwork(hidden_size, num_classes)
        self.ensemble_classifier = nn.ModuleList([
                    modules.ClassificationNetwork(hidden_size, num_classes) 
                    for _ in range(n_classifiers-1)])
        # oracle
        self.sensor_oracle = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn_oracle = modules.CoreNetwork(hidden_size, hidden_size)
        self.classifier_oracle = modules.ClassificationNetwork(hidden_size, num_classes)

    def forward(self, x, l_t_prev, h_t_prev, h_t_oracle_prev, last=False):
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)

        g_t_oracle = self.sensor_oracle(x, l_t_prev)
        h_t_oracle = self.rnn_oracle(g_t_oracle, h_t_oracle_prev)

        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            c_log_pi_ensemble_list = []
            for classifier in self.ensemble_classifier:
                c_log_pi_i = classifier(h_t.detach())
                c_log_pi_ensemble_list.append(c_log_pi_i)
                
            if c_log_pi_ensemble_list == []:
                c_log_pi_ensemble = None
            else: 
                c_log_pi_ensemble = torch.stack(c_log_pi_ensemble_list)

            c_log_pi = self.classifier(h_t)
            c_log_pi_oracle = self.classifier_oracle(h_t_oracle.detach())
            return h_t, l_t, b_t, c_log_pi, c_log_pi_ensemble, log_pi, c_log_pi_oracle

        return h_t, l_t, b_t, log_pi, h_t_oracle, 


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes, n_classifiers, 
    ):
        """Constructor.

        Args:
          g: size of the square patches in the glimpses extracted by the retina.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
        """
        super().__init__()

        self.std = std

        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = modules.CoreNetwork(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)
        self.classifier = modules.ClassificationNetwork(hidden_size, num_classes)
        self.ensemble_classifier = nn.ModuleList([
                    modules.ClassificationNetwork(hidden_size, num_classes) 
                    for _ in range(n_classifiers-1)])

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """Run RAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)

        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            c_log_pi_ensemble_list = []
            for classifier in self.ensemble_classifier:
                c_log_pi_i = classifier(h_t.detach())
                c_log_pi_ensemble_list.append(c_log_pi_i)
                
            if c_log_pi_ensemble_list == []:
                c_log_pi_ensemble = None
            else: 
                c_log_pi_ensemble = torch.stack(c_log_pi_ensemble_list)

            c_log_pi = self.classifier(h_t)

            # c_log_pi = self.classifier(h_t)
            return h_t, l_t, b_t, c_log_pi, c_log_pi_ensemble, log_pi

        return h_t, l_t, b_t, log_pi

class EarlyStoppingRecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes, n_classifiers,
    ):
        """Constructor.

        Args:
          g: size of the square patches in the glimpses extracted by the retina.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
        """
        super().__init__()

        self.std = std

        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = modules.CoreNetwork(hidden_size, hidden_size)
        # self.planner = modules.Plannernetwork(hidden_size, 2)
        self.locator = modules.LocationNetworkEarlyStopping(hidden_size, 3, std)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)
        self.classifier = modules.ClassificationNetwork(hidden_size, num_classes)
        # self.ensemble_classifier = nn.ModuleList([
        #             modules.ClassificationNetwork(hidden_size, num_classes) 
        #             for _ in range(n_classifiers-1)])
        # modules.ClassificationNetwork(hidden_size, num_classes)

    def forward(self, x, l_t_prev, h_t_prev):
        """Run RAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 3). The location vector
                containing the glimpse coordinates [x, y, p_stop] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 3). The location vector
                containing the glimpse coordinates [x, y, p_stop] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        # p_log_pi, action_or_stop = self.planner(h_t)
        l_t, l_log_pi, s_pi = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        # c_log_pi_list = []
        # for classifier in self.ensemble_classifier:
        #     c_log_pi = classifier(h_t)
        #     c_log_pi_list.append(c_log_pi)
            
        # c_log_pi = torch.stack(c_log_pi_list)
        c_log_pi = self.classifier(h_t)
        
        # return h_t, p_log_pi, action_or_stop, l_t, l_log_pi, b_t, c_log_ps
        return h_t, l_t, l_log_pi, s_pi, b_t, c_log_pi


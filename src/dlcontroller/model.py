"""
Originally implemented by: https://github.com/pranz24/pytorch-soft-actor-critic
Check LICENSE for details
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, max_horizon):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs * max_horizon + num_actions * max_horizon, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, max_horizon)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs * max_horizon + num_actions * max_horizon, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, max_horizon)

        self.apply(weights_init_)

    def forward(self, state, action):
        # print('action shape in Q-network: ', action.shape)
        xu = torch.cat([state, action], 1)

        # Q1 architecture
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2 architecture
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class SequenceQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, max_horizon, input_sequence_length, output_sequence_length):
        super(SequenceQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs * input_sequence_length + num_actions * input_sequence_length, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_sequence_length)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs * input_sequence_length + num_actions * input_sequence_length, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, output_sequence_length)

        self.apply(weights_init_)

    def forward(self, state, action):
        # print('action shape in Q-network: ', action.shape)
        xu = torch.cat([state, action], 1)

        # Q1 architecture
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2 architecture
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                 is_recurrent=False, init_mu=0, init_std=1, min_std=1e-6):
        super(GaussianPolicy, self).__init__()
        self.recurrent = is_recurrent
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.max_horizon = 100
        self.min_horizon = 1
        self.init_mu = init_mu
        self.init_std = init_std
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(num_inputs * (self.max_horizon + 1), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        """ length of the trajectory """
        self.horizon_mean = nn.Linear(hidden_dim, 1)  # for horizon
        self.horizon_log_std = nn.Linear(hidden_dim, 1)  # for horizon

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        horizon_mean = self.horizon_mean(x)
        horizon_log_std = self.horizon_log_std(x)
        horizon_log_std = torch.clamp(horizon_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, horizon_mean, horizon_log_std

    def sample(self, state):
        mean, log_std, horizon_mean, horizon_log_std = self.forward(state)
        std = log_std.exp()
        horizon_std = horizon_log_std.exp()
        # print('horizon mean: ', horizon_mean)
        horizon_mean = torch.tanh(horizon_mean)  # * self.action_scale + self.action_bias
        horizon_normal = LogNormal(horizon_mean, horizon_std)
        h_t = horizon_normal.rsample()

        h_t = torch.tanh(h_t)
        horizon = h_t * self.max_horizon + self.min_horizon
        horizon = horizon.int().flatten()
        actions_stack, means_stack, log_probs_stack = [], [], []

        for mu, sigma, h in zip(mean, std, horizon):
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
            normal = Normal(mu, sigma)
            action_cat, mean_cat, log_prob_cat = [], [], []
            for i in range(h):
                """ sampling 'h' times """
                x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
                y_t = torch.tanh(x_t)  # muscle activations
                action = y_t * self.action_scale + self.action_bias
                action_cat.append(action.flatten())
                log_prob = normal.log_prob(x_t)
                # Enforcing Action Bound
                log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
                log_prob = log_prob.sum(1, keepdim=True)
                log_prob_cat.append(log_prob)
                mean = torch.tanh(mu) * self.action_scale + self.action_bias
                mean_cat.append(mean.flatten())
            action_cat = torch.stack(action_cat).unsqueeze(0)
            action_cat = torch.reshape(action_cat, (action_cat.shape[0], action_cat.shape[1] * action_cat.shape[
                2]))  # reshaping to (batch_size, horizon * num_actions)
            action_pad = (0, 202 - action_cat.shape[1])  # (padding_left, padding_right)
            action_cat = F.pad(action_cat, action_pad, "constant", value=0)
            mean_cat = torch.stack(mean_cat).unsqueeze(0)  # .flatten()
            mean_cat = torch.reshape(mean_cat, (mean_cat.shape[0], mean_cat.shape[1] * mean_cat.shape[2]))
            mean_pad = (0, 202 - mean_cat.shape[1])  # (padding_left, padding_right)
            mean_cat = F.pad(mean_cat, mean_pad, "constant", value=0)
            log_prob_cat = torch.stack(log_prob_cat).unsqueeze(0)
            log_prob_cat = torch.reshape(log_prob_cat,
                                         (log_prob_cat.shape[0], log_prob_cat.shape[1] * log_prob_cat.shape[2]))
            log_prob_pad = (0, 202 - log_prob_cat.shape[1])
            log_prob_cat = F.pad(log_prob_cat, log_prob_pad, "constant", value=0)
            actions_stack.append(action_cat.squeeze(0))
            means_stack.append(mean_cat.squeeze(0))
            log_probs_stack.append(log_prob_cat.squeeze(0))
        actions_stack = torch.stack(actions_stack)
        means_stack = torch.stack(means_stack)
        log_probs_stack = torch.stack(log_probs_stack).sum(1, keepdim=True)
        return actions_stack, log_probs_stack, means_stack

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class ControllerPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, output_sequence_length, action_space=None):
        super(ControllerPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.softplus = nn.Softplus()
        self.state_vector = nn.Linear(hidden_dim, output_sequence_length)
        self.state_time = nn.Linear(hidden_dim, output_sequence_length)
        self.mean = nn.Linear(hidden_dim, num_actions)  # muscle activations
        self.activation_time = nn.Linear(hidden_dim, output_sequence_length)
        self.action_noise = torch.Tensor(num_actions)
        self.action_time_noise = torch.Tensor(output_sequence_length)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        activation_time = torch.cumsum(self.softplus(self.activation_time(x)), dim=0)
        state_time = torch.cumsum(self.softplus(self.state_time(x)), dim=0)
        state_vector = self.state_vector(x)
        return mean,  state_vector, activation_time, state_time

    def sample(self, state):
        mean, state_vector, activation_time, state_time = self.forward(state)
        action_noise = self.action_noise.normal_(0., std=0.1)
        action_noise = action_noise.clamp(-0.25, 0.25)
        action = mean + action_noise
        action_time_noise = self.action_time_noise.normal_(0., std=1)
        action_time_noise = action_time_noise.clamp(0.1, 3.0)
        activation_time = activation_time + action_time_noise
        return action, torch.tensor(0.), mean, state_vector, activation_time, state_time

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.action_noise = self.action_noise.to(device)
        self.action_time_noise = self.action_time_noise.to(device)
        return super(ControllerPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class DynamicsNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, max_horizon):
        super(DynamicsNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs * max_horizon + num_actions * max_horizon, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_inputs * max_horizon)

        self.apply(weights_init_)

    def forward(self, state, action):
        # print('action shape in dynamics network: ', action.shape)
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

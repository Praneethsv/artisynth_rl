"""
Originally implemented by: https://github.com/pranz24/pytorch-soft-actor-critic
Check LICENSE for details
"""
import os

import torch
import torch.nn.functional as F
import torch.nn.functional
from torch.optim import Adam
from dlcontroller.utils import soft_update, hard_update
import torch.nn as nn
import numpy as np

from dlcontroller.model import GaussianPolicy, QNetwork, DeterministicPolicy, DynamicsNetwork, ControllerPolicy
from dlcontroller.rnn_model import NormalVAE
from common.utilities_pytorch import ExponentialLRWithMin



class DlController:
    def __init__(self, num_inputs, action_space, args):
        super(DlController, self).__init__()

        self.optims = dict()
        self.models = dict()
        self.lr_schedulers = []

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.updates_count = 0

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, max_horizon=101).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.lr_schedulers.append(ExponentialLRWithMin(self.critic_optim, args.lr_gamma, min=args.lr_min))
        self.optims['critic_optim'] = self.critic_optim
        self.models['critic'] = self.critic

        # Dynamics network
        self.dynamic = DynamicsNetwork(num_inputs, action_space.shape[0], args.hidden_size, max_horizon=101).to(device=self.device)
        self.dynamic_optim = Adam(self.dynamic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, max_horizon=101).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                self.lr_schedulers.append(ExponentialLRWithMin(self.alpha_optim, args.lr_gamma, min=args.lr_min))
                self.models['alpha_optim'] = self.alpha_optim

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        elif self.policy_type == "Controller":
            self.policy = ControllerPolicy(num_inputs, action_space.shape[0], output_sequence_length=args.static_horizon,
                                        hidden_dim=args.hidden_size, action_space=action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        elif self.policy_type == "VAE":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                self.lr_schedulers.append(ExponentialLRWithMin(self.alpha_optim, args.lr_gamma, min=args.lr_min))
                self.models['alpha_optim'] = self.alpha_optim
            self.policy = NormalVAE(input_sequence_length=num_inputs * args.static_horizon,
                                    latent_size=action_space.shape[0],
                                    output_sequence_length=action_space.shape[0] * args.static_horizon,
                                    rnn_type='rnn', hidden_size=args.lstm_hidden_size, bidirectional=False).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.lr_schedulers.append(ExponentialLRWithMin(self.policy_optim, args.lr_gamma, min=args.lr_min))
        self.optims['policy_optim'] = self.policy_optim
        self.models['policy'] = self.policy

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval_mode:
            action, _, _, state_vector, activation_time, state_time = self.policy.sample(state)
        else:
            _, _, action, state_vector, activation_time, state_time = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        activation_time = activation_time.detach().cpu().numpy()[0]
        activation_time = np.round(activation_time, 2)
        return action, activation_time, state_vector, state_time

    def update_parameters(self, memory, batch_size):
        print("updating parameters ... ")
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)

        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        """ dynamics network """
        pred_next_state = self.dynamic(state_batch, action_batch)
        dynamic_loss = F.l1_loss(pred_next_state, next_state_batch)

        action_pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, action_pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.dynamic_optim.zero_grad()
        dynamic_loss.backward()
        self.dynamic_optim.step()

        for scheduler in self.lr_schedulers:
            scheduler.step()

        return policy_loss.item(), dynamic_loss.item()

    # Save model parameters
    def save_model(self, filepath, global_episodes):
        model_states = dict()
        optim_states = dict()

        # neural models
        for key, value in self.models.items():
            model_states.update({key: value.state_dict()})

        # optimizers' states
        for key, value in self.optims.items():
            optim_states.update({key: value.state_dict()})

        states = {'global_episode_num': global_episodes,  # to avoid saving right after loading
                  'model_states': model_states,
                  'optim_states': optim_states}
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

        # make sure KeyboardInterrupt exceptions don't mess up the model saving process
        while True:
            try:
                with open(filepath, 'wb+') as f:
                    torch.save(states, f)
                break
            except KeyboardInterrupt:
                pass

    # Load model parameters
    def load_model(self, filepath, load_optim=True):
        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f)

        for key in self.models.keys():
            self.models[key].load_state_dict(checkpoint['model_states'][key], strict=False)

        hard_update(self.critic_target, self.critic)

        if load_optim:
            # todo: make sure learning rate loads from optim_state
            self.lr_schedulers.clear()
            for key in self.optims.keys():
                self.optims[key].load_state_dict(checkpoint['optim_states'][key])

        return checkpoint['global_episode_num']


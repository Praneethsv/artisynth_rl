import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from models import actor, critic
from hindsight_utils import sync_networks, sync_grads
from rb import replay_buffer
from normalizer import normalizer
from original_her import her_sampler
from tensorboardX import SummaryWriter
from src.artisynth.envs.point_model_env import PointModel

"""
ddpg with HER (MPI-version) for point model

"""

width_scale = 1.0


class point_ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        muscle_labels = ["m" + str(i) for i in np.array(range(args.num_muscles))]
        env = PointModel(verbose=0, success_thres=args.success_threshold, dof_observation=args.dob, include_follow=False, port=args.port,
                         muscle_labels=muscle_labels,
                        )
        self.env = env
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name,
                                           self.args.exp_name)  # self.args.env_name
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        writer = SummaryWriter('./ddpg+her_logs/{}/{}'.format(self.args.env_name, self.args.exp_name),
                               comment="-ddpg+her")
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions, mb_widths, mb_distances, mb_infos, mb_amps = [], [], [], [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_widths, ep_distances, ep_infos, ep_amps = [], [], [], [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset(it=epoch, goal='random', num_samples=0)
                    # print(observation)
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    target_width = observation['target_width']
                    distance = observation['distance']

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):  # self.env_params['max_timesteps']
                        # self.env.render()
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g, target_width, distance)
                            # print('input_tensor', input_tensor)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        width_new = observation_new['target_width']
                        # print(width_new)
                        distance_new = observation_new['distance']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        ep_widths.append(target_width)
                        ep_distances.append(distance)
                        ep_infos.append(np.linalg.norm(info['grip_velp']))
                        ep_amps.append(info['movement_amplitude'])
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                        target_width = width_new
                        distance = distance_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_widths.append(target_width)  # svp
                    ep_distances.append(distance)  # svp
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_widths.append(ep_widths)
                    mb_distances.append(ep_distances)
                    mb_infos.append(ep_infos)
                    mb_amps.append(ep_amps)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_widths = np.array(mb_widths)
                mb_distances = np.array(mb_distances)
                mb_infos = np.expand_dims(mb_infos, axis=-1)
                mb_amps = np.expand_dims(mb_amps, axis=-1)
                # print(mb_widths)
                # store the episodes
                self.buffer.store_episode(
                    [mb_obs, mb_ag, mb_g, mb_actions, mb_widths, mb_distances, mb_infos, mb_amps])  # svp
                self._update_normalizer(
                    [mb_obs, mb_ag, mb_g, mb_actions, mb_widths, mb_distances, mb_infos, mb_amps])  # svp
                critic_losses = []
                for _ in range(self.args.n_batches):
                    # train the network
                    critic_loss = self._update_network()
                    critic_losses.append(critic_loss)
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate, reward, distance = self._eval_agent()

            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, reward is: {}'.format(datetime.now(), epoch,
                                                                                              success_rate, reward))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor_network.state_dict()], \
                           self.model_path + '/model.pt')
                writer.add_scalar("Success Rate", success_rate, epoch)
                writer.add_scalar("rewards", reward, epoch)
                writer.add_scalar("distances", distance, epoch)
                writer.add_scalar("critic loss", np.mean(np.array(critic_losses)), epoch)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g, target_width, distance):
        # print(type(target_width))
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        target_width = np.array([target_width * width_scale])
        distance = np.array([distance])
        # concatenate the stuffs
        # print(obs_norm.shape)
        # print(g_norm.shape)
        # print(target_width.shape)
        # print(distance.shape)
        inputs = np.concatenate([obs_norm, g_norm, target_width, distance])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, mb_widths, mb_distances, mb_infos, mb_amps = episode_batch

        # print(type(mb_obs))
        mb_obs_next = mb_obs[:, 1:, :]  # remove the first one
        mb_ag_next = mb_ag[:, 1:, :]  # remove the first one
        mb_w_next = mb_widths[:, 1:]  # remove the first
        mb_d_next = mb_distances[:, 1:]

        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'w': mb_widths,
                       'd': mb_distances,
                       'actions': mb_actions,
                       'grip_velp': mb_infos,
                       'movement_amplitude': mb_amps,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'w_next': mb_w_next,
                       'd_next': mb_d_next
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # print('rewards are:', transitions['r'])
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _preproc_wd(self, w, d):
        w = width_scale * w
        d = d
        return w, d

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        critic_losses = []
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, w, d, w_next, d_next = transitions['obs'], transitions['obs_next'], transitions['g'], transitions[
            'target_width'], transitions['distance'] \
            , transitions['w_next'], transitions['d_next']

        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['w'], transitions['d'] = self._preproc_wd(w, d)
        # print(transitions['w'])
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        transitions['w_next'], transitions['d_next'] = self._preproc_wd(w_next, d_next)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm, w, d], axis=1)

        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm, w_next, d_next], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        # print(inputs_next_norm)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        # print('real actions', actions_real)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        return critic_loss.cpu().data.numpy()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate, total_distances, total_rewards = [], [], []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate, rewards, distances = [], [], []
            observation = self.env.reset(it=0, goal='linear', num_samples=0)
            obs = observation['observation']
            g = observation['desired_goal']
            target_width = observation['target_width']
            distance = observation['distance']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g, target_width, distance)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
                rewards.append(reward)
                distances.append(np.linalg.norm(g - observation_new['achieved_goal']))
            total_distances.append(distances)
            total_rewards.append(rewards)
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        total_rewards = np.array(total_rewards)
        total_distances = np.array(total_distances)
        local_success_rate = np.mean(total_success_rate[:, -1])
        local_rewards = np.sum(total_rewards[:, -1])
        local_distances = np.sum(total_distances[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_reward = MPI.COMM_WORLD.allreduce(local_rewards, op=MPI.SUM)
        global_distance = MPI.COMM_WORLD.allreduce(local_distances, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_reward / MPI.COMM_WORLD.Get_size(), \
               global_distance / MPI.COMM_WORLD.Get_size()

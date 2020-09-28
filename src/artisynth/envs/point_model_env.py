import os
import time
import logging

import torch

import gym
from gym import error, spaces
from gym.utils import seeding

from common import constants as c
from common.net import Net

import numpy as np


NUM_TESTING_STEPS = 50

LOW_EXCITATION = 0
HIGH_EXCITATION = 1

logger = logging.getLogger()

COMPS = ['ref_pos',
         'follow_pos']


def success_threshold():
    width = (np.random.randint(low=2, high=6) / 10.0,)
    return np.asarray(width)


def eval_success_threshold(episode):
    width = (0.2 + episode / 10.0,)
    return np.asarray(width)


def get_excitations_time_labels(length):
    labels = []
    excitation_time_str = "excitationsTime"
    for i in range(length):
        labels.append(excitation_time_str + str(i))
    return labels


def get_excitations_labels(length):
    labels = []
    excitation_str = "M"
    for i in range(length):
        labels.append(excitation_str + str(i))
    return labels


class PointModelEnv(gym.Env):

    def __init__(self, ip, port, wait_action, eval_mode=False, reset_step=20, include_time=False,
                 init_artisynth=False, include_target_width=False, include_distance=False, include_target_state=True,
                 include_excitations=False, include_current=True, include_velocity=False,
                 ):
        super(PointModelEnv).__init__()
        self.sock = None

        self.net = Net(ip, port)
        self.episode_counter = 0
        self.reset_step = reset_step
        self.eval_mode = eval_mode

        self.wait_action = wait_action
        logger.info("Inside the constructor of point_model")
        self.include_excitations = include_excitations
        self.include_current = include_current
        self.include_velocity = include_velocity
        self.include_target_width = include_target_width
        self.include_target_state = include_target_state
        self.include_distance = include_distance
        self.include_time = include_time

        if init_artisynth:
            logger.info('Running artisynth')
            self.run_artisynth(ip, port)

        self.action_size = self.get_action_size()
        print('action size', self.action_size)
        obs = self.reset(None)
        logger.info('State array size: {}'.format(obs.shape))
        self.obs_size = obs.shape[0]

        self.observation_space = spaces.Box(low=-0.2, high=+0.2,
                                            shape=[self.obs_size], dtype=np.float32)

        self.observation_space.shape = (self.obs_size,)

        # init action space
        self.action_space = spaces.Box(low=LOW_EXCITATION, high=HIGH_EXCITATION,
                                       shape=(self.action_size,),
                                       dtype=np.float32)

    def run_artisynth(self, ip, port):
        if ip != 'localhost':
            raise NotImplementedError

        command = 'artisynth -model artisynth.models.lumbarSpine.RlLumbarSpine ' + \
                  '[ -port {} ] -play -noTimeline'. \
                      format(port)
        command_list = command.split(' ')

        import subprocess
        FNULL = open(os.devnull, 'w')
        subprocess.Popen(command_list, stdout=FNULL, stderr=subprocess.STDOUT)
        time.sleep(3)

    def get_state_dict(self):
        self.net.send(message_type=c.GET_STATE_STR)
        state_dict = self.net.receive_message(c.STATE_STR, retry_type=c.GET_STATE_STR)
        return state_dict

    def get_state_size(self):
        self.net.send(message_type=c.GET_STATE_SIZE_STR)
        rec_dict = self.net.receive_message(c.STATE_SIZE_STR, c.GET_STATE_SIZE_STR)
        logger.info('State size: {}'.format(rec_dict[c.STATE_SIZE_STR]))
        return rec_dict[c.STATE_SIZE_STR]

    def get_action_size(self):
        self.net.send(message_type=c.GET_ACTION_SIZE_STR)
        rec_dict = self.net.receive_message(c.ACTION_SIZE_STR, c.GET_ACTION_SIZE_STR)
        logger.info('Action size: {}'.format(rec_dict[c.ACTION_SIZE_STR]))
        return rec_dict[c.ACTION_SIZE_STR]

    def state_dict2tensor(self, state):
        return torch.tensor(self.state_dic_to_array(state))

    def get_state_tensor(self):
        state_dict = self.get_state_dict()
        return self.state_dict2tensor(state_dict)

    def take_action(self, action):
        print('activations: ', action)
        self.net.send({'excitations': action}, message_type='setExcitations')
        rec_dict = self.net.receive_message(c.EXCITATIONS_DONE_STR)
        return rec_dict[c.EXCITATIONS_DONE_STR]

    def take_action_time(self, action_time):
        self.net.send({'excitationsTime': action_time}, message_type='setExcitationsTime')
        rec_dict = self.net.receive_message(msg_type=c.EXCITATIONS_TIME_DONE_STR)
        print('rec_dict in take_action_time is: ', rec_dict)
        return rec_dict[c.EXCITATIONS_TIME_DONE_STR]

    def augment_action_time(self, action_time):
        excitations_time_labels = get_excitations_time_labels(int(self.action_size / 2))
        return dict(zip(excitations_time_labels, action_time))

    def augment_action(self, actions):
        excitation_labels = get_excitations_labels(self.action_size)
        return dict(zip(excitation_labels, actions))

    def step(self, action):

        action = self.augment_action(action)
        self.net.send({'excitations': action}, message_type='setExcitations')
        # self.net.receive_message()
        state = self.get_state_dict()
        if state is not None:
            new_ref_pos = np.asarray(state['ref_pos'])
            new_follower_pos = np.asarray(state['follow_pos'])
            new_follower_vel = np.asarray(state['follow_vel'])[:3]
            distance = self.calculate_distance(new_ref_pos, new_follower_pos)
            if self.prev_distance is not None:
                reward, done = self.compute_reward(distance, self.prev_distance, new_follower_vel)
            else:
                reward, done = (0, False)
            self.prev_distance = distance
            if done:
                self.log('Achieved done state', verbose=0)
            self.log('Reward: ' + str(reward), verbose=1, same_line=True)
            state_arr = self.state_json_to_array(state, self.success_thres)

            info = {'amplitude': distance,
                    'velocity': np.linalg.norm(new_follower_vel),
                    'distance': np.linalg.norm(new_ref_pos),
                    'width': self.success_thres}

        return state_arr, reward, done, info

    def controller_step(self, actions, actions_time):
        info = {}
        done = False
        logger.debug('action:{}'.format(actions))
        exc_time_set = self.take_action_time(self.augment_action_time(actions_time))
        print('time set: ', exc_time_set)
        if exc_time_set:
            print("taking actions ... ")
            actions = np.clip(actions, LOW_EXCITATION, HIGH_EXCITATION)
            actions = self.augment_action(actions)
            excs_done = self.take_action(actions)
        print('excitations filled: ', excs_done)
        if excs_done:
            state = self.get_state_dict()
        if not state:
            return None, 0, False, {}
        state_arr = self.state_dic_to_array(state, self.success_thres)
        info['state_time'] = state[c.TIME]
        if self.episode_counter >= self.reset_step:
            done = True
        return state_arr, done, info

    @staticmethod
    def parse_state(state_dict: dict):
        state = {'ref_pos': np.array(
            [float(s) for s in state_dict['ref_pos'].split(" ")]),
            'follow_pos': np.array(
                [float(s) for s in state_dict['follow_pos'].split(" ")])
        }
        return state

    def set_state(self, state):
        self._set_state(state[:3], state[4:])

    def _set_state(self, ref_pos, follower_pos):
        self.ref_pos = ref_pos
        self.follower_pos = follower_pos

    def state_json_to_array(self, state_dict: dict, success_thres):
        state_arr = np.asarray(state_dict['ref_pos'])
        assert self.include_follow
        if self.include_follow:
            state_arr = np.concatenate((state_arr, state_dict['follow_pos']))
        if self.follow_velocity_include:
            state_arr = np.concatenate((state_arr, state_dict['follow_vel'][:3]))
        if self.include_width:
            state_arr = np.concatenate((state_arr, success_thres))
        if self.include_distance:
            state_arr = np.concatenate((state_arr, ))
        return state_arr

    def state_dic_to_array(self, js, success_thres):
        logger.debug('state json: %s', str(js))

        observation_vector = []

        if self.include_current:
            observation_vector.extend(js[COMPS[0]])
        if self.include_target_state:
            observation_vector.extend(js[COMPS[1]])
        if self.include_distance:
            observation_vector.extend(js['distanceError'])
        if self.include_distance:
            observation_vector.extend(js['time'])
        """ TODO: include time """

        return np.asarray(observation_vector)

    def get_state_dict(self):
        self.net.send(message_type=c.GET_STATE_STR)
        state_dict = self.net.receive_message(c.STATE_STR, retry_type=c.GET_STATE_STR)
        return state_dict

    def reset(self, episode):
        self.net.send(message_type=c.RESET_STR)
        self.ref_pos = None
        self.prev_distance = None

        state_dict = self.get_state_dict()
        print('state_dict from the server: ', state_dict)
        logger.log(msg=['Target: %s',
                        (['{:.4f}'.format(x) for x in state_dict[COMPS[1]]])], level=15)
        if episode is None:
            self.success_thres = success_threshold()
        else:
            self.success_thres = eval_success_threshold(episode)
        state = self.state_dic_to_array(state_dict, success_thres=self.success_thres)
        print('state after reset: ', state)
        return state

    def render(self, mode='human', close=False):
        # our environment does not need rendering
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    def configure(self, *args, **kwargs):
        pass

    def compute_speed_reward(self, ref_pos, prev_follow_pos, new_follow_pos, velocity):
        return - self.calculate_distance(ref_pos, new_follow_pos) + velocity

    def compute_reward(self, new_dist, prev_dist, velocity):
        if new_dist < self.success_thres:
            # achieved done state
            return 5, True  # (-new_dist + np.linalg.norm(velocity))
        else:
            if prev_dist - new_dist > 0:
                return 1 / self.agent.episode_step, False
            else:
                return -1, False
from src.artisynth.envs.point_model2d_env import PointModel2dEnv
from src.common.utilities import begin_time
from src.common import config as config
from src.common import constants as c
from src.common.net import Net

import numpy as np

def success_threshold():

    return

class PointModel(PointModel2dEnv):

    def __init__(self, verbose, muscle_labels, dof_observation, success_thres, port,
                 include_follow, log_to_file=True, log_file='log', agent=None,
                 ip='localhost'):
        super(PointModel).__init__(muscle_labels=muscle_labels, dof_observation=dof_observation, success_thres=success_thres,
                                   ip=ip, port=port, include_follow=include_follow, verbose=verbose, agent=agent, log_to_file=log_to_file,
                                   log_file=log_file)
        self.success_thres = success_threshold()  # change this

    def step(self, action):

        action = self.augment_action(action)
        self.net.send({'excitations': action}, message_type='setExcitations')
        state = self.get_state_dict()
        if state is not None:
            new_ref_pos = np.asarray(state['ref_pos'])
            new_follower_pos = np.asarray(state['follow_pos'])
            new_follower_vel = np.asarray(state['follow_vel'])[:2]
            distance = self.calculate_distance(new_ref_pos, new_follower_pos)
            if self.prev_distance is not None:
                reward, done = self.compute_reward(distance, self.prev_distance)
            else:
                reward, done = (0, False)
            self.prev_distance = distance
            if done:
                self.log('Achieved done state', verbose=0)
            self.log('Reward: ' + str(reward), verbose=1, same_line=True)

            state_arr = self.state_json_to_array(state)
            info = {'distance': distance,
                    'velocity': np.linalg.norm(new_follower_vel)}

        return state_arr, reward, done, info

    def set_state(self, state):

        self._set_state(state[:3], state[4:])

    def get_state_dict(self):
        self.net.send(message_type=c.GET_STATE_STR)
        state_dict = self.net.receive_message(c.STATE_STR, retry_type=c.GET_STATE_STR)
        return state_dict

    def compute_speed_reward(self, ref_pos, prev_follow_pos, new_follow_pos, velocity):

        return - self.calculate_distance(ref_pos, new_follow_pos) + velocity

    def compute_reward(self, new_dist, prev_dist):
        if new_dist < self.success_thres:
            # achieved done state
            return 0, True
        else:
            return -1, False

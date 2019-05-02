from src.artisynth.envs.point_model2d_env import PointModel2dEnv
from src.common.utilities import begin_time
from src.common import config as config
from src.common import constants as c
from src.common.net import Net
from rl.core import Processor
import numpy as np


def success_threshold():

    return np.random.randint(low=1, high=8) / 10.0


class PointModelEnv(PointModel2dEnv):

    def __init__(self, verbose, muscle_labels, dof_observation, success_thres, port,
                 include_follow=True, follow_velocity_include=True, log_to_file=True, log_file='log', agent=None,
                 ip='localhost'):
        super(PointModelEnv).__init__()
        self.net = Net(ip, port)

        self.verbose = verbose
        self.success_thres = success_thres
        self.ref_pos = None

        self.action_space = type(self).ActionSpace(muscle_labels)
        self.observation_space = type(self).ObservationSpace(
            dof_observation)  # np.random.rand(dof_observation)
        self.log_to_file = log_to_file
        self.log_file_name = log_file
        if log_to_file:
            self.logfile, path = type(self).create_log_file(log_file)
            self.log('Logging into file: ' + path, verbose=1)
        self.agent = agent
        self.include_follow = include_follow
        self.port = port
        self.prev_distance = None
        self.muscle_labels = muscle_labels
        self.follow_velocity_include = follow_velocity_include

    def step(self, action):

        action = self.augment_action(action)
        self.net.send({'excitations': action}, message_type='setExcitations')
        state = self.get_state_dict()
        if state is not None:
            new_ref_pos = np.asarray(state['ref_pos'])
            new_follower_pos = np.asarray(state['follow_pos'])
            new_follower_vel = np.asarray(state['follow_vel'])[:3]
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

    def state_json_to_array(self, state_dict: dict):
        state_arr = np.asarray(state_dict['ref_pos'])
        assert self.include_follow
        if self.include_follow:
            state_arr = np.concatenate((state_arr, state_dict['follow_pos']))
        if self.follow_velocity_include:
            state_arr = np.concatenate((state_arr, state_dict['follow_vel'][:3]))
        return state_arr

    def get_state_dict(self):
        self.net.send(message_type=c.GET_STATE_STR)
        state_dict = self.net.receive_message(c.STATE_STR, retry_type=c.GET_STATE_STR)
        return state_dict

    def reset(self):
        self.net.send(message_type=c.RESET_STR)
        self.ref_pos = None
        self.prev_distance = None
        self.log('Reset', verbose=0)
        state_dict = self.get_state_dict()
        state = self.state_json_to_array(state_dict)
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

    def compute_reward(self, new_dist, prev_dist):
        if new_dist < self.success_thres:
            # achieved done state
            return 1 / new_dist, True
        else:
            if prev_dist - new_dist > 0:
                return 1 / self.agent.episode_step, False
            else:
                return -1, False


class PointModelProcessor(Processor):
    """Abstract base class for implementing processors.
        A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
        be necessary if your agent has different requirements with respect to the form of the
        observations, actions, and rewards of the environment. By implementing a custom processor,
        you can effectively translate between the two without having to change the underlaying
        implementation of the agent or environment.
        Do not use this abstract base class directly but instead use one of the concrete implementations
        or write your own.
        """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.
        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.
        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        # print('observation', observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.
        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []
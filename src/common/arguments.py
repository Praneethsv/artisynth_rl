import argparse

import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.conflict_handler = 'resolve'

    parser.add_argument('--ip', type=str, default='localhost',
                        help='IP of server')
    parser.add_argument('--verbose', type=int, default='20',
                        help='Verbosity level')
    parser.add_argument('--init_artisynth', type=str2bool, default=True,
                        help='run environment with GUI.')
    parser.add_argument('--experiment_name', default='UnknownModel',
                        help='Name of the experiment, for logging purposes.')
    parser.add_argument('--env', default='Point2PointEnv-v0',
                        help='environment to train on')
    parser.add_argument('--model-name', default='testModel',
                        help='Name of the RL model being trained for logging purposes.')
    parser.add_argument('--load-path', default=None,
                        help='Path to load the trained model.')
    parser.add_argument('--port', type=int, default=4545,
                        help='port to run the server on (default: 4545)')
    parser.add_argument('--visdom-port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--episode-log-interval', type=int, default=1,
                        help='log interval for episodes (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--wait-action', type=float, default=0.0,
                        help='Wait (seconds) for action to take place and environment to stabilize.')
    parser.add_argument('--episodic', action='store_true', default=False,
                        help='Whether task is episodic.')
    parser.add_argument('--test', type=str2bool, default=False,
                        help='Only evaluate a trained model.')
    parser.add_argument('--use_wandb', type=str2bool, default=False,
                        help='Use wandb for train logging.')
    parser.add_argument('--use_tensorboard', default=False,
                        help='use tensorboard for logging')
    parser.add_argument('--resume-wandb', action='store_true', default=False,
                        help='Resume the wandb training log.')
    parser.add_argument('--reset-step', type=int, default=-1,
                        help='Reset envs every n iters.')
    parser.add_argument('--hidden-layer-size', type=int, default=64,
                        help='Number of neurons in all hidden layers.')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save interval, one save per n updates')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='eval interval, one eval per n updates')
    parser.add_argument('--episode_log_interval', type=int, default=1,
                        help='log interval for episodes')
    parser.add_argument('--eval_episode', type=int, default=5,
                        help='Number of episodes to evaluate')

    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps (default: 5)')
    parser.add_argument('--num-steps-eval', type=int, default=5,
                        help='number of forward steps in evaluation (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=10e3,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--goal_threshold', type=float, default=0.1,
                        help='Difference between real and target which is considered as success '
                             'when reaching a goal')
    parser.add_argument('--goal_reward', type=float, default=0,
                        help='The reward to give if goal was reached.')
    parser.add_argument('--zero_excitations_on_reset', type=str2bool, default=True,
                        help='Reset all muscle excitations to zero after each reset.')
    parser.add_argument('--reward', type=str, default="intermediate",
                        help='type of reward to use: Step or Intermediate')
    parser.add_argument('--reward_type', type=str, default="dense",
                        help='sparse or dense reward to choose for the algorithm')
    parser.add_argument('--wait_action', type=float, default=0.0,
                        help='Wait time for action\'s impact to be perceived in the environment '
                             '(sec)')
    parser.add_argument('--incremental_actions', type=str2bool, default=False,
                        help='Treat actions as increment/decrements to the current excitations.')

    # environment observation space
    parser.add_argument('--include_target_width', type=str2bool, default=False,
                        help='Include target width in the state space')
    parser.add_argument('--include_velocity', type=str2bool, default=False,
                        help='Include velocity in the state space')
    parser.add_argument('--include_perceptual_width', type=str2bool, default=False,
                        help='Include perceptual width in the state space')
    parser.add_argument('--include_time', type=str2bool, default=False,
                        help='Include the duration of excitation in the state space')
    parser.add_argument('--reset_step', type=int, default=1e10, help='Reset envs every n iters.')
    parser.add_argument('--include_current_state', type=str2bool, default=True,
                        help='Include the current position/rotation of the model in the state.')
    parser.add_argument('--include_target_state', type=str2bool, default=True,
                        help='Include the target position/rotation/velocity of the model in the state.')
    parser.add_argument('--include_current_excitations', type=str2bool, default=False,
                        help='Include the current excitations of actuators in the state.')
    parser.add_argument('--include_distance_error', type=str2bool, default=False,
                        help='Include the error to the target distance in the state.')
    parser.add_argument('--include_displacement', type=str2bool, default=False,
                        help='Include the distance travelled by the agent.')
    parser.add_argument('--include_acceleration', type=str2bool, default=False,
                        help='Include the distance travelled by the agent.')

    return parser

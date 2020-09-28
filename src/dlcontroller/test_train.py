"""
Originally implemented by: https://github.com/pranz24/pytorch-soft-actor-critic
Check LICENSE for details
"""
import datetime
import itertools
from tensorboardX import SummaryWriter
import os
import pickle
import numpy as np

from dlcontroller.replay_memory import ReplayMemory
from common.config import setup_logger
from common.utilities import get_lr_pytorch



def train(env, agent, args, configs, memory=None, global_episodes=0):
    logger = setup_logger()

    # TesnorboardX
    if args.use_tensorboard:
        writer = SummaryWriter(
            logdir='{}/{}_SAC_{}_{}_{}'.format(configs.tensorboard_log_directory,
                                               datetime.datetime.now().strftime(
                                                   "%Y-%m-%d_%H-%M-%S"), args.env,
                                               args.policy,
                                               "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = memory or ReplayMemory(args.replay_size)

    # Training Loop
    global_steps = 0

    for global_episodes in itertools.count(start=global_episodes, step=1):
        episode_steps = 0
        done = False
        reset_states = []
        print("Inside train of the test_train file:")
        state = env.reset(None)

        if args.policy == "VAE" or "Controller":
            for _ in range(args.static_horizon):
                reset_states.append(state)

            state = np.array(reset_states)

        policy_loss_total = 0
        dynamic_loss_total = 0

        while not done:
            action, actions_time, _, _, = agent.select_action(state)
            action = np.squeeze(action)

            if len(memory) > args.batch_size and global_steps > args.start_steps:
                # print('updating', len(memory), global_steps)
                for i in range(args.updates_per_step):  # Number of updates per step in environment
                    policy_loss, dynamic_loss = \
                        agent.update_parameters(memory, args.batch_size)  # update all parameters

                    policy_loss_total += policy_loss
                    dynamic_loss_total += dynamic_loss

            next_state, done, info = env.controller_step(action.flatten(), actions_time.flatten())  # Step
            episode_steps += 1
            global_steps += 1
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.reset_step else float(not done)
            memory.push(state.flatten(), action.flatten(), next_state.flatten(), mask)  # Append transition to memory
            state = next_state
        # end of episode

        # The following values are a bit off for the first episode as we have no updates
        # for len(memory) < batch_size
        policy_loss_total /= (episode_steps * args.updates_per_step)
        dynamic_loss_total /= episode_steps


        if global_episodes % args.episode_log_interval == 0:
            print("Episode: {}, total num steps: {}, ep steps: {}".format(
                global_episodes, global_steps, episode_steps))
            if args.use_tensorboard:
                writer.add_scalar('loss/policy', policy_loss_total, global_episodes)
                writer.add_scalar('loss/dynamic_loss', dynamic_loss_total, global_episodes)
            if args.use_wandb:
                import wandb
                wandb.log({
                    'loss/policy': policy_loss_total,
                    'loss/dynamics': dynamic_loss_total,
                    'lr': get_lr_pytorch(agent.policy_optim)},
                    step=global_episodes)

        if global_episodes % args.eval_interval == args.eval_interval - 1:
            avg_reward, infos = _test(env, agent, configs, args.eval_episode, args)
            if args.use_tensorboard:
                writer.add_scalar('eval/avg_reward', avg_reward, global_episodes)
                for key, val in infos.items():
                    writer.add_scalar(f'eval/{key}', val, global_episodes)
            if args.use_wandb:
                import wandb
                wandb.log({'eval/avg_reward': avg_reward}, step=global_episodes)
                for key, val in infos.items():
                    wandb.log({f'eval/{key}': val}, step=global_episodes)

        if global_episodes % args.save_interval == args.save_interval - 1:
            test_save_path = os.path.join(configs.trained_directory, 'test_file')

            # TODO: update the following hack by saving file temp and copy to destination
            with open(test_save_path, 'w') as test_file:
                test_file.write("This is just to make sure we have enough disk space to fully save "
                                "the file not to screw up the agent or the memory! " * 1000)
            
            agent_save_path = os.path.join(configs.trained_directory, 'agent')
            agent.global_episode = global_episodes + 1
            # torch.save(agent, agent_save_path)
            agent.save_model(agent_save_path, global_episodes)
            logger.info(f'model saved: {agent_save_path}')

            # memory_path = os.path.join(configs.trained_directory, 'memory')
            # pickle.dump(memory, open(memory_path, 'wb'))
            # logger.info(f'memory saved: {memory_path}')
            print('------------------')

        # if global_steps > args.num_steps:  # end of training
        #     break

        if global_episodes > args.num_episodes:
            break

    env.close()


def padarray(array, size, pad_boolean=False):
    t = size - array.ravel().shape[0]
    if not pad_boolean:
        a = np.pad(array.ravel(), pad_width=(0, t), mode='constant', constant_values=0)
    else:
        a = np.pad(array.ravel(), pad_width=(0, t), mode='constant', constant_values=str(0))
    return a         # a.reshape(int(size/num_inputs), num_inputs)


def test(env, agent, configs, args):
    logger = setup_logger()
    env.seed(args.seed)
    avg_reward, infos = _test(env, agent, configs, args.test_episode, args)
    logger.info('Test trial complete. Writing results...')

    results_path = args.load_path + '_test_results'

    if args.env[0:6] == 'JawEnv':
        from artisynth.envs.jaw_env import write_infos, calculate_convex_hull, \
            maximum_occlusal_force
        write_infos(infos, results_path)

        # Derived metrics
        maximum_occlusal_force(env, results_path)
        calculate_convex_hull(results_path)

    logger.info(f'results written to: {results_path}')
    env.close()


# def velocity_plot(velocities, dirname, fname):
#     import matplotlib.pyplot as plt
#     import os
#     # vels = [0]
#     vels = velocities
#     print(vels)
#     widths = [0.2, 0.3,
#               0.4, 0.5, 0.6,
#               0.7, 0.8, 0.9,
#               1.0, 1.1, 1.2,
#               1.3, 1.4, 1.5,
#               1.6, 1.7, 1.8,
#               1.9, 2.0]
#
#     plt.plot(widths, vels)
#     plt.ylabel('velocities')
#     plt.xlabel('widths')
#     if not os.path.isdir(dirname):
#         os.mkdir(dirname)
#     # plt.savefig('vel_profs')
#     # plt.savefig(os.path.join(dirname, fname))
#     # plt.clf()
#     plt.savefig('vels_trend')


def _test(env, agent, configs, episodes, args):
    avg_reward = 0.
    infos = {}
    fitts_infos = []
    avg_vels = []
    num_inputs = env.observation_space.shape[0]
    max_horizon = 101
    for episode_count in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        episode_iter_count = 0
        info_episode_avg = {}
        info_episode_final = {}

        fitts_info = {}
        ep_vels = []
        ep_dists = []
        ep_times = []

        info_episode_all = {}

        state = padarray(state, num_inputs * max_horizon)

        while not done:
            action, actions_time, _, _ = agent.select_action(state, eval_mode=True)
            print('action values are: ', action)
            next_state, reward, done, info = env.controller_step(action, actions_time)
            episode_reward += np.mean(reward)
            state = next_state
            episode_iter_count += 1
            ep_vels.append(info['velocity'])
            ep_dists.append(info['distanceError'])
            ep_times.append(info['activationTime'])
            # ep_widths.append(info['targetWidth'])

            # if info['time'] < 0.6:  # sleepSeconds as hard coded on the java side
            #     # print('time:', info['time'])
            #     done = False
            #     continue

            for key, val in info.items():
                # keep the average and last item for scalars
                if isinstance(val, float):
                    info_episode_avg['avg_' + key] = info_episode_avg.get('avg_' + key, 0) + val
                    info_episode_final['final_' + key] = val

                if 'all_' + key not in info_episode_all.keys():
                    info_episode_all['all_' + key] = list()
                info_episode_all['all_' + key].append(val)

        episode_reward /= episode_iter_count
        avg_reward += episode_reward
        fitts_info['d'] = ep_dists
        fitts_info['v'] = ep_vels
        fitts_info['MT'] = ep_times[len(ep_vels) - 1]
        # fitts_info['n'] = len(ep_vels)
        fitts_info['W'] = info['targetWidth']
        # fitts_info['D'] = info['targetAmplitude']
        fitts_episode_str = "Fitts Params: "
        episode_print_str = "{}/{} reward:{:.3f}".format(episode_count, episodes, episode_reward)
        fname_str = "velocity_profile_D:{}-W:{}".format(args.dist_idx, str(fitts_info['W']).replace('.', '_'))
        # dirname_str = "vel_profiles_D_{}_width_varying".format(args.dist_idx)
        # dir_str = "vel_profiles_DI_{}_width_varying".format(args.dist_idx)

        for key in info_episode_avg.keys():
            info_episode_avg[key] /= episode_iter_count
            infos[key] = infos.get(key, 0) + info_episode_avg[key]
            episode_print_str += "  {}:{:.3f}".format(key, info_episode_avg[key])
        avg_vels.append(info_episode_avg['avg_velocity'])
        for key in info_episode_final.keys():
            infos[key] = infos.get(key, 0) + info_episode_final[key]
            episode_print_str += "  {}:{:.3f}".format(key, info_episode_final[key])

        for key in fitts_info.keys():
            fitts_episode_str += "  {}:{}".format(key, fitts_info[key])

        for key in info_episode_all.keys():
            # create a list of lists (first list is episodes, second is iterations of the episodes)
            if key not in infos.keys():
                infos[key] = list()
            infos[key].append(info_episode_all[key])

        print(episode_print_str)
        print(fitts_episode_str)

        # velocity_plot(ep_vels, dirname='run', fname=fname_str)
        fitts_infos.append(fitts_info)

    avg_reward /= episodes
    print_str = f'Test #Episodes: {episodes}, avg_reward: {round(avg_reward, 3)}'
    # print(avg_vels)
    # velocity_plot(avg_vels, dirname='vel_trend', fname='velocity_vs_width')
    for key in infos.keys():
        if not isinstance(infos[key], list):  # don't do this for the info_episode_all values
            infos[key] /= episodes
            print_str += f' {key}:{infos[key]:.3f}'

    with open(configs.trained_directory + '/' + 'fitts.p', 'wb') as f:
        pickle.dump(fitts_infos, f)

    print("----------------------------------------")
    print(print_str)
    print("----------------------------------------")

    return avg_reward, infos


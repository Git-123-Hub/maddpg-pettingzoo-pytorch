import argparse

import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_reference_v2, simple_spread_v2, \
    simple_tag_v2, simple_world_comm_v2, simple_push_v2, simple_speaker_listener_v3
import matplotlib.pyplot as plt

from MADDPG import MADDPG


def get_env(name, N=2, max_cycles=25, continuous_actions=True):
    name_map = {  # env function, env name
        'adversary': [simple_adversary_v2.parallel_env, 'simple_adversary_v2'],
        'crypto': [simple_crypto_v2.parallel_env, 'simple_crypto_v2'],
        'push': [simple_push_v2.parallel_env, 'simple_push_v2'],
        'reference': [simple_reference_v2.parallel_env, 'simple_reference_v2'],
        'speaker': [simple_speaker_listener_v3.parallel_env, 'simple_speaker_listener_v2'],
        'spread': [simple_spread_v2.parallel_env, 'simple_spread_v2'],
        'tag': [simple_tag_v2.parallel_env, 'simple_tag_v2'],
        'comm': [simple_world_comm_v2.parallel_env, 'simple_world_comm_v2'],
    }
    env_fn, full_name = name_map[name]
    return env_fn(N=N, max_cycles=max_cycles, continuous_actions=continuous_actions), full_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='adversary', help='name of the env',
                        choices=['adversary', 'crypto', 'push', 'reference', 'speaker', 'spread', 'tag',
                                 'comm'])
    parser.add_argument('--episode-num', type=int, default=500,
                        help='total episode num during training procedure')
    parser.add_argument('--learn-interval', type=int, default=10, help='steps interval between learning time')
    parser.add_argument('--update-interval', type=int, default=30,
                        help='step interval of updating target network')
    parser.add_argument('--tau', type=float, default=0.01, help='soft update parameter')
    args = parser.parse_args()
    # todo: use continuous action or discrete action
    # todo: option on creating env
    env, env_name = get_env(args.env_name)
    env.reset()
    maddpg = MADDPG(env)

    step = 0
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            actions = maddpg.select_action(states)
            next_states, rewards, dones, infos = env.step(actions)
            maddpg.add(states, actions, rewards, next_states, dones)
            states = next_states

            for agent, reward in rewards.items():  # update reward
                agent_reward[agent] += reward

            if step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn()

            if step % args.update_interval == 0:  # update target network every few steps
                maddpg.update_target(args.tau)

        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent, reward in agent_reward.items():
            episode_rewards[agent][episode] = reward
            message += f'{agent}: {reward:>4f}; '
        print(message)

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'maddpg solve {env_name}'
    ax.set_title(title)
    plt.savefig(title)

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MADDPG import MADDPG
from util import get_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='adversary', help='name of the env',
                        choices=['adversary', 'crypto', 'push', 'reference', 'speaker', 'spread', 'tag',
                                 'comm'])
    parser.add_argument('folder', type=str, default='1', help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    args = parser.parse_args()

    env, env_name = get_env(args.env_name)
    model_dir = os.path.join('./results', env_name, args.folder)
    assert os.path.exists(model_dir)
    gif_dir = os.path.join(model_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env.reset()
    # maddpg = MADDPG(env)
    # maddpg.load(os.path.join(model_dir, 'model.pt'))
    maddpg = MADDPG.init_from_file(env, os.path.join(model_dir, 'model.pt'))

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        while env.agents:  # interact with the env for an episode
            actions = maddpg.select_action(states, explore=False)
            # actions['adversary_0'] = env.action_space('adversary_0').sample()
            # actions['agent_0'] = env.action_space('agent_0').sample()
            next_states, rewards, dones, infos = env.step(actions)
            states = next_states
            frame_list.append(Image.fromarray(env.render(mode='rgb_array')))

            for agent, reward in rewards.items():  # update reward
                agent_reward[agent] += reward

        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent, reward in agent_reward.items():
            episode_rewards[agent][episode] = reward
            message += f'{agent}: {reward:>4f}; '
        print(message)
        # save gif
        frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                           save_all=True, append_images=frame_list[1:], duration=1, loop=0)

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve {env_name} {total_files - 3}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))

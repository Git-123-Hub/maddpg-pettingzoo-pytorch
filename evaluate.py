import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from MADDPG import MADDPG
from main import get_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='adversary', help='name of the env',
                        choices=['adversary', 'crypto', 'push', 'reference', 'speaker', 'spread', 'tag',
                                 'comm'])
    parser.add_argument('model_name', type=str, default='model1', help='filename of the model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='path where all the training results are saved')
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    args = parser.parse_args()

    env, env_name = get_env(args.env_name)
    if args.model_path is None:
        model_path = os.path.join('./results', env_name)
    else:
        model_path = args.model_path
    assert os.path.exists(model_path)

    env.reset()
    maddpg = MADDPG(env)
    maddpg.load(os.path.join(model_path, f'{args.model_name}.pt'))

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            actions = maddpg.select_action(states, explore=False)
            next_states, rewards, dones, infos = env.step(actions)
            states = next_states
            env.render()

            for agent, reward in rewards.items():  # update reward
                agent_reward[agent] += reward

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

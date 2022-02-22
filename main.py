import argparse

from pettingzoo.mpe import simple_adversary_v2

from MADDPG import MADDPG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='adversary', help='name of the env',
                        choices=['adversary', 'crypto', 'push', 'reference', 'speaker', 'spread', 'tag',
                                 'comm'])
    args = parser.parse_args()
    # todo: get env form args.env_name
    # todo: use continuous action or discrete action
    # todo: option on creating env
    env = simple_adversary_v2.parallel_env(N=2, max_cycles=25, continuous_actions=True)
    states = env.reset()
    maddpg = MADDPG(env)
    while env.agents:
        # actions = {agent: policy(observations[agent], agent) for agent in parallel_env.agents}
        # actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        actions = maddpg.select_action(states)
        next_states, rewards, dones, infos = env.step(actions)
        maddpg.add(states, actions, rewards, next_states, dones)
        states = next_states

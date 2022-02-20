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
    env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True)
    env.reset()
    maddpg = MADDPG(env)

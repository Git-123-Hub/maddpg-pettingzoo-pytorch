from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv

from Agent import Agent


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent that can interact with env"""

    def __init__(self, env: SimpleEnv):
        # create agent according to all the agents of the env
        action_info = {}
        for agent in env.agents:
            action_info[agent] = []
            action_info[agent].append(env.observation_space(agent).shape[0])
            action_info[agent].append(env.action_space(agent).shape[0])
        print(action_info)

        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for _, val in action_info.items())

        # create Agent(actor-critic) for each agent
        self.agents = {}
        for agent in env.agents:
            self.agents[agent] = Agent(*action_info[agent], global_obs_act_dim)
            # print(self.agents[agent].actor)
            # print(self.agents[agent].critic)

    def update(self):
        pass

    def update_target(self, tau):
        for agent in self.agents:
            agent.update_target(tau)

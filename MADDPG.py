from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv

from Agent import Agent
from ReplayBuffer import ReplayBuffer


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent that can interact with env"""

    def __init__(self, env: SimpleEnv):
        # create agent according to all the agents of the env
        action_info = {}
        for agent in env.agents:
            action_info[agent] = []
            action_info[agent].append(env.observation_space(agent).shape[0])
            action_info[agent].append(env.action_space(agent).shape[0])

        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for _, val in action_info.items())

        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        for agent in env.agents:
            self.agents[agent] = Agent(*action_info[agent], global_obs_act_dim)
            # todo: option for replay buffer
            self.buffers[agent] = ReplayBuffer(10000, 256)

    def learn(self):
        pass

    def update_target(self, tau):
        for agent in self.agents:
            agent.update_target(tau)

    def select_action(self, states):
        actions = {}
        for agent, state in states.items():
            print(agent, state)
            actions[agent] = self.agents[agent].act(state)
        return actions

    def add(self, states, actions, rewards, next_states, dones):
        for agent in states.keys():
            state = states[agent]
            action = actions[agent]
            reward = rewards[agent]
            next_state = next_states[agent]
            done = dones[agent]
            self.buffers[agent].add(state, action, reward, next_state, done)

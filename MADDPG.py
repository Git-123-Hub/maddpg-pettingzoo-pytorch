import os

import numpy as np
import torch
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
import torch.nn.functional as F

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
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(list(self.buffers.values())[0])
        if total_num <= 256:  # only start to learn when there are enough experiences to sample
            return
        # sample from all the replay buffer using the same index
        # todo: how to use batch size
        indices = np.random.choice(total_num, size=256, replace=False)
        samples = {}
        state_list, act_list, next_state_list, next_act_list = [], [], [], []
        for agent, buffer in self.buffers.items():
            transitions = buffer.sample(indices)
            samples[agent] = transitions
            state_list.append(transitions[0])
            act_list.append(transitions[1])
            next_state_list.append(transitions[3])
            # calculate next_action using target_network and next_state
            next_act_list.append(self.agents[agent].act(transitions[3], target=True, ndarray=False))

        # critic input all the states and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        critic_in = torch.cat(state_list + act_list, 1)  # torch.Size([batch_size, global_obs_act_dim])
        target_critic_in = torch.cat(next_state_list + next_act_list, 1)

        # update all agents
        for name, agent in self.agents.items():
            # update critic
            states, actions, rewards, next_states, dones = samples[name]
            # todo: specify gamma
            target_value = rewards + 0.99 * agent.critic_value(target_critic_in, target=True) * (1 - dones)
            critic_value = agent.critic_value(critic_in)  # tensor with the length of batch_size
            critic_loss = F.mse_loss(target_value.detach(), critic_value, reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # calculate action using actor
            action = agent.act(states, ndarray=False)
            act_list = []
            for agent_name in self.agents.keys():  # loop over all the agents
                if agent_name == name:  # action of the current agent is calculate using its actor
                    act_list.append(action)
                else:  # action of other agents is from the samples
                    act_list.append(samples[agent_name][1])
            critic_in = torch.cat(state_list + act_list, 1)
            actor_loss = -agent.critic_value(critic_in).mean()
            actor_loss += (action ** 2).mean() * 1e-3  # todo: how to calculate loss of actor
            # actor_loss = -(action * critic_value).mean()
            agent.update_actor(actor_loss)

    def scale_noise(self, scale):
        for agent in self.agents.values():
            agent.noise.scale = scale

    def update_target(self, tau):
        for agent in self.agents.values():
            agent.update_target(tau)

    def select_action(self, states, explore=True):
        actions = {}
        for agent, state in states.items():
            actions[agent] = self.agents[agent].act(state, explore=explore)
        return actions

    def add(self, states, actions, rewards, next_states, dones):
        for agent in states.keys():
            state = states[agent]
            action = actions[agent]
            reward = rewards[agent]
            next_state = next_states[agent]
            done = dones[agent]
            self.buffers[agent].add(state, action, reward, next_state, done)

    def save(self, folder):
        """save actor parameter of all agents"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(folder, 'model.pt')
        )

    def load(self, filename):
        data = torch.load(filename)
        for agent_name, agent in self.agents.items():
            agent.actor.load_state_dict(data[agent_name])

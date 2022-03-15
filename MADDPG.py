import os

import numpy as np
import torch
from gym.spaces import Box
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
import torch.nn.functional as F

from Agent import Agent
from ReplayBuffer import ReplayBuffer
from util import setup_logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, env: SimpleEnv, capacity, batch_size, actor_lr, critic_lr):
        continuous = False  # default: discrete
        if isinstance(env.action_space('agent_0'), Box):
            continuous = True

        # create agent according to all the agents of the env
        action_info = {}
        for agent in env.agents:
            action_info[agent] = []
            action_info[agent].append(env.observation_space(agent).shape[0])
            if continuous:
                action_info[agent].append(env.action_space(agent).shape[0])
            else:
                action_info[agent].append(env.action_space(agent).n)

        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for val in action_info.values())

        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        self.batch_size = batch_size
        for agent in env.agents:
            self.agents[agent] = Agent(*action_info[agent], global_obs_act_dim, actor_lr, critic_lr,
                                       continuous)
            self.buffers[agent] = ReplayBuffer(capacity, self.batch_size)
        self.logger = setup_logger('maddpg.log', 'maddpg')

    @classmethod
    def init_from_file(cls, env, file, continuous):
        """init maddpg using the model saved in `file`"""
        # get env dimension info to initialize actor network
        action_info = {}
        for agent in env.agents:
            action_info[agent] = []
            action_info[agent].append(env.observation_space(agent).shape[0])
            if continuous:
                action_info[agent].append(env.action_space(agent).shape[0])
            else:
                action_info[agent].append(env.action_space(agent).n)

        instance = cls(env, 0, 0, 0, 0)
        # only actor are needed when evaluate
        instance.agents = {}
        for agent in env.agents:
            instance.agents[agent] = Agent(*action_info[agent], 1, 0, 0, continuous)
        data = torch.load(file)
        for agent_name, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_name])
        return instance

    def learn(self, gamma):
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(list(self.buffers.values())[0])
        if total_num <= self.batch_size:  # only start to learn when there are enough experiences to sample
            return

        # sample from all the replay buffer using the same index
        indices = np.random.choice(total_num, size=self.batch_size, replace=False)
        samples = {}
        state_list, act_list, next_state_list, next_act_list = [], [], [], []
        for agent, buffer in self.buffers.items():
            transitions = buffer.sample(indices)
            samples[agent] = transitions
            state_list.append(transitions[0])
            act_list.append(transitions[1])
            next_state_list.append(transitions[3])
            # calculate next_action using target_network and next_state
            next_act_list.append(self.agents[agent].target_action(transitions[3]))

        # update all agents
        for cur_agent_name, agent in self.agents.items():
            # update critic
            states, actions, rewards, next_states, dones = samples[cur_agent_name]
            critic_value = agent.critic_value(state_list, act_list)  # tensor with the length of batch_size

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(next_state_list, next_act_list)
            target_value = rewards + gamma * next_target_critic_value * (1 - dones)

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            action_list = []
            for agent_name in self.agents.keys():  # loop over all the agents
                if agent_name == cur_agent_name:  # action of the current agent is calculated using its actor
                    # todo: try with noise
                    action = agent.action(states, explore=False)  # NOTE that NO noise
                else:  # action of other agents is from the samples
                    action = samples[agent_name][1]
                action_list.append(action)
            actor_loss = -agent.critic_value(state_list, action_list).mean()
            agent.update_actor(actor_loss)
            self.logger.info(f'{cur_agent_name}: critic loss: {critic_loss.item()}, '
                             f'actor loss: {actor_loss.item()}')

    def scale_noise(self, scale):
        for agent in self.agents.values():
            agent.noise.scale = scale
            agent.noise_scale = scale

    def reset_noise(self):
        for agent in self.agents.values():
            agent.noise.reset()

    def update_target(self, tau):
        for agent in self.agents.values():
            agent.update_target(tau)

    def select_action(self, states, explore=True):
        actions = {}
        for agent, state in states.items():
            action = self.agents[agent].action(state, explore=explore)  # torch.Size([1, action_size])
            actions[agent] = action.squeeze(0).detach().numpy()
            self.logger.info(f'{agent} action: {actions[agent]}')
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

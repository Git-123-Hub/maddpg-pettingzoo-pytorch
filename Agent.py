from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import gumbel_softmax, one_hot
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, continuous):
        if continuous:  # use last_layer to constrain output  # todo: change to Tanh
            self.actor = MLPNetwork(obs_dim, act_dim, last_layer=nn.Sigmoid())
        else:  # the actor output will be logit of each action
            self.actor = MLPNetwork(obs_dim, act_dim)

        # critic input all the states and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.noise = OUNoise(act_dim)  # todo: option on ou-noise
        self.noise_scale = 1
        self.continuous = continuous

    # todo: change method `action` and `target_action` to continuous domains
    def action(self, state, *, explore):
        # this method is called in the following two cases:
        # a) interact with the environment, where input is a numpy.ndarray
        # NOTE that the output is a tensor, you have to convert it to ndarray before input to the environment
        # b) when update actor, calculate action using actor and states,
        # which is sampled from replay buffer with size: torch.Size([batch_size, state_dim])

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0)  # torch.Size([1, state_size])
        action = self.actor(state)  # torch.Size([batch_size, action_size])
        if explore:
            action = gumbel_softmax(action, hard=True)
            # if hard=True, the returned samples will be discretized as one-hot vectors
        else:
            # choose action with the biggest actor_output(logit)
            max_index = action.max(dim=1)[1]
            action = one_hot(max_index)
        return action  # onehot tensor with size: torch.Size([batch_size, action_size])

    def target_action(self, state):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        action = self.target_actor(state)  # torch.Size([batch_size, action_size])
        # NOTE that I didn't use noise during this procedure
        # so I just choose action with the biggest actor_output(logit)
        max_index = action.max(dim=1)[1]
        action = one_hot(max_index).detach()
        return action  # onehot tensor with size: torch.Size([batch_size, action_size])

    def _critic_value(self, x, *, target=False):
        if target:
            return self.target_critic(x).squeeze(1)  # tensor with length of batch_size
        else:
            return self.critic(x).squeeze(1)

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        return self._critic_value(torch.cat(state_list + act_list, 1), target=False)

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        return self._critic_value(torch.cat(state_list + act_list, 1), target=True)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """
            copy the parameters of `from_network` to `to_network` with a proportion of tau
            i.e. update `to_network` parameter with tau of `from_network`
            """
            for to_p, from_p in zip(to_network.parameters(), from_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        soft_update(self.actor, self.target_actor)
        soft_update(self.critic, self.target_critic)


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU(), last_layer=None):
        super(MLPNetwork, self).__init__()

        modules = [
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ]
        if last_layer is not None:
            modules.append(last_layer)
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

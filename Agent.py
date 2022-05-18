from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
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

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    # todo: change method `action` and `target_action` to continuous domains
    def action(self, obs):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        action = self.gumbel_softmax(logits)
        # if model_out:
        #     return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        action = self.gumbel_softmax(logits)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


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

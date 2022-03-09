from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, continuous):
        if continuous:  # use last_layer to constrain output  # todo: change to Tanh
            self.actor = MLPNetwork(obs_dim, act_dim, last_layer=nn.Sigmoid())
        else:  # the actor output will be logit of each action
            self.actor = MLPNetwork(obs_dim, act_dim)

        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.noise = OUNoise(act_dim)  # todo: option on ou-noise
        self.noise_scale = 1
        self.continuous = continuous

    # todo: more method on act, target act
    def act(self, state, *, target=False, ndarray=True, explore=True):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0)  # torch.Size([1, state_size])
        if target:  # use target network to get target action
            action = self.target_actor(state)  # torch.Size([1, action_size])
        else:
            action = self.actor(state)  # torch.Size([1, action_size])

        if self.continuous:  # actor directly output action
            if explore:
                action += torch.from_numpy(self.noise.noise()).unsqueeze(0)
                # action += torch.tensor(np.random.uniform(-1, 1)).unsqueeze(0) * self.noise_scale
            action.clip_(0, 1)
        else:  # actor output prob of each action
            if explore:
                action = gumbel_softmax(action, hard=True)
                # if hard=True, the returned samples will be discretized as one-hot vectors
            else:
                # choose action with the biggest actor_output(logit)， convert logit to onehot
                action = action.squeeze(0)
                action_index = action.argmax().item()
                action.fill_(0)
                action[action_index] = 1

        action = action.detach().squeeze(0)  # tensor of length: action_size
        if ndarray:
            return action.numpy()  # ndarray of length: action_size
        else:
            return action

    def critic_value(self, x, *, target=False):
        if target:
            return self.target_critic(x).squeeze(1)  # tensor with length of batch_size
        else:
            return self.critic(x).squeeze(1)

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

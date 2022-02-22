from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, lr=0.005):
        # todo: add lr to args
        self.actor = MLPNetwork(obs_dim, act_dim, last_layer=nn.Sigmoid())
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0)  # torch.Size([1, state_size])
        action = self.actor(state)  # torch.Size([1, action_size])
        return action.detach().squeeze(0).numpy()  # ndarray of length: action_size

    def get_value(self, obs, act):
        pass

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

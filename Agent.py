from copy import deepcopy

from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""
    def __init__(self, obs_dim, act_dim, global_obs_dim, lr):
        self.actor = (obs_dim, act_dim)
        self.critic = (global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def act(self, obs):
        pass

    def get_value(self, obs, act):
        pass

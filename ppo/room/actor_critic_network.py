
import torch.nn as nn
from torch.distributions import Categorical

class SeparateActorCritic(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(SeparateActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Sigmoid(),
        )
        
    def forward(self, state_observations):
        state_value = self.critic(state_observations)

        action_probabilities = self.actor(state_observations)
        action_distribution = Categorical(action_probabilities)
        
        return action_distribution, state_value
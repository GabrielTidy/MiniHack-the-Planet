
import torch.nn as nn
from torch.distributions import Categorical

class SeparateActorCritic(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(SeparateActorCritic, self).__init__()
        
        #critic neural network - outputs the value of the inputted state
        #uses ReLU activation functions for non-linearity
        self.critic = nn.Sequential(
            nn.Linear(num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        #actor neural network - outputs weights for each available action given the inputted state
        #the action weights are used to determine probabilities for each action (the higher the weight, the higher the probability)
        #uses ReLU activation functions for non-linearity, as well as a Sigmoid function at the final layer to ensure each weight can be used as a probability
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

        #the action probabilities from the actor network are used to create a distribution from which actions can be sampled
        action_probabilities = self.actor(state_observations)
        action_distribution = Categorical(action_probabilities)
        
        return action_distribution, state_value
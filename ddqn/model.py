import torch.nn as nn
import torch


class DuelingDQN(nn.Module):
    """
    A basic implementation of a neural network to predict Q-value function
    """

    def __init__(self, observation_space, action_space):
        """
        Initialise the Q-network with 2 streams
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super(DuelingDQN,self).__init__()

        self.features = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.value= nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage= nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        

    def forward(self, state):
        state_features = self.features(state) 
        state_values = self.value(state_features)
        action_advantages = self.advantage(state_features)
        q_values = state_values + (action_advantages -action_advantages.mean())
        return q_values


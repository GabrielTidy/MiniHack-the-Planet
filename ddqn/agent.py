from gym import spaces
import numpy as np
import torch
from model import DuelingDQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the network learning algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingDQN(observation_space,action_space).to(self.device)
        self.target_model = DuelingDQN(observation_space,action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr)
        self.loss= torch.nn.MSELoss()
        self.r_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_ddqn = use_double_dqn

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        
        states,actions,rewards,n_states,dones = self.r_buffer.sample(self.batch_size)
        
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards =torch.from_numpy(rewards).float().to(self.device)
        n_states = torch.from_numpy(n_states).float().to(self.device)


        self.optimizer.zero_grad()
        
        idx = np.arange(self.batch_size)
        q_policy_s = (self.model(states)[idx,actions]).double()
        q_target_s_ = (self.target_model(n_states)).detach()
        max_a = torch.argmax(self.model(n_states),1)

        q_target_s_[dones]=0.0
        q_target_s = (rewards+(self.gamma*q_target_s_[idx,max_a])).double()
        
        loss = self.loss(q_target_s,q_policy_s).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        
        self.target_model.load_state_dict(self.model.state_dict())
        

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        
        
        s = np.copy(state)
        s= torch.from_numpy(s).float().to(self.device)
        s = s.unsqueeze(0)
        q_out = self.model(s).to('cpu').detach().numpy()
        return np.argmax(q_out)
    
        
    def save_model(self,fname):
        torch.save(self.model.state_dict(),fname)
#         torch.save(self.target_model.state_dict(), f'Target network {env_name}')
        
    def load_model(self,fname):
        self.model.load_state_dict(torch.load(fname))
#         self.target_model.load_state_dict(torch.load(f'Target network {env_name}'))

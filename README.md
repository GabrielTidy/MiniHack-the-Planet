# MiniHack-the-Planet

This project requires the use of nethack and minihack, which can be installed following the steps at:
https://github.com/facebookresearch/nle
https://github.com/facebookresearch/minihack

MiniHack is a sandbox framework for easily designing rich and diverse environments for Reinforcement Learning (RL). Based on the game of NetHack, MiniHack uses the NetHack Learning Environment (NLE) to communicate with the game and to provide a convenient interface for customly created RL training and test environments of varying complexity. In this assignment our task is to create an agent that can navigate the Quest-Hard dungeon in Minihack and accrue as many points as possible. The two methods we consider for this task are dueling deep Q-Network (DQN) and Proximal Policy Optimization (PPO).

The following python packages are required to be installed for both the DDQN and PPO agents:
  - numpy
  - gym

# DDQN  AGENT

In order to run and compile the DQN Agent follow the following steps

1) Clone the ddqn GitHub repository,
2) Open the folder in your terminal or preferred IDE and run the Deep-Q Learning agent, 
3) When you initially run the main files you will be prompted of dependency errors,
4) Install all the dependences required and also refer to the minihack documentation for guidelines


# PPO AGENT

The PPO files require the following python packages to be installed:
  - os
  - datetime
  - matplotlib
  - torch

Once these have been installed, run the files from the terminal while in the PPO folder using the command
  python3 ppo_[name].py
  
with [name] replaced by the type of environment you wish to train on. These include:
  - room
  - lava
  - maze
  - quest_easy

Running each file will train an agent in the corresponding environment using PPO, and will display and save a recording (gif) of the final run once the target reward has been reached.

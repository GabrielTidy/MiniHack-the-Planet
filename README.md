# MiniHack-the-Planet

This project requires the use of nethack and minihack, which can be installed following the steps at:
- https://github.com/facebookresearch/nle
- https://github.com/facebookresearch/minihack


The following python packages are required to be installed for both the 3DQN and PPO agents:
  - numpy
  - gym (version 0.23.0 for 3DQN and version 0.25.2 for PPO)
  - matplotlib
  - torch

# 3DQN AGENT

In order to run and compile the 3DQN Agent follow the following steps (assuming all dependencies are installed). Note that version of gym required is 0.23.0.

1) Clone the ddqn GitHub repository,
2) Open the folder in your terminal or preferred IDE and run the command 'python train_minihack.py' 


# PPO AGENT

The PPO files require the following python packages to be installed:
  - os
  - datetime
  - tensorboard
  - gym (version 0.25.2)
  
Once these have been installed, run the files from the terminal while in the PPO folder using the command
  python3 ppo_[name].py
  
with [name] replaced by the type of environment you wish to train on. These are:
  - MiniHack-Room-15x15-v0 (python3 ppo_room.py)
  - MiniHack-MazeWalk-15x15-v0 (python3 ppo_maze.py)
  - MiniHack-LavaCross-Levitate-Potion-Inv-v0 (python3 ppo_lava.py)
  - MiniHack-Quest-Easy-v0 (python3 ppo_quest_easy.py)
  - MiniHack-Quest-Hard-v0 (python3 ppo_quest_hard.py)

Running each file will train an agent in the corresponding environment using PPO, and will display and save a recording (gif) of the final run once the target reward has been reached.

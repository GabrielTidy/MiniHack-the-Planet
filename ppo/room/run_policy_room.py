
import numpy as np
import gym
import torch
from argparse import ArgumentParser
from actor_critic_network import SeparateActorCritic
from nle import nethack
import minihack
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


### Hyperparameters ###

#nethack minhack environment
ENVIRONMENT_NAME = "MiniHack-Room-Random-5x5-v0"
OBSERVATION_KEYS = ("chars_crop", "pixel",)
ACTION_KEYS = (nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W)
PENALTY_STEP = -0.01
PENALTY_TIME = 0
REWARD_WIN = 1
REWARD_LOSE = -1

#whether to sample actions or always choose action with max probability
DETERMINISTIC = True

### End Hyperparameters ###

def goal_distance(chars):
    player_pos = np.where(chars == 64)
    if player_pos[0].shape[0] == 0:
        return 0.0
    player_pos = [player_pos[0][0], player_pos[1][0]]
    goal_pos = np.where(chars == 62)
    if goal_pos[0].shape[0] == 0:
        return 0.0
    goal_pos = [goal_pos[0][0], goal_pos[1][0]]
    return np.sqrt((player_pos[0]-goal_pos[0])**2 + (player_pos[1]-goal_pos[1])**2) * 0.1

def positions(chars):
    player_pos = np.where(chars == 64)
    player_pos = [player_pos[0][0], player_pos[1][0]]
    goal_pos = np.where(chars == 62)
    goal_pos = [goal_pos[0][0], goal_pos[1][0]]
    return np.concatenate((player_pos, goal_pos))

#environment observation processing
def process_observations(obs):
    # game_map = obs["chars_crop"].flatten()
    # return game_map
    return positions(obs["chars_crop"])
    
  

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p", "--policy", required=True, help="Policy checkpoint file to load")
    parser.add_argument("-r", "--record", default=False, help="If True, saves a video recording of the run (default=False")
    args = parser.parse_args()

    env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=OBSERVATION_KEYS, 
    reward_win=REWARD_WIN, reward_lose=REWARD_LOSE, penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP)

    # env.seed(42)

    raw_state = env.reset()
    state = process_observations(raw_state)

    num_observations = state.shape[0]
    num_actions = env.action_space.n

    actor_critic = SeparateActorCritic(num_observations, num_actions)
    actor_critic.load_state_dict(torch.load(args.policy))

    PIXEL_HISTORY = [raw_state["pixel"]]

    # env.render()

    reward_sum = 0
    num_timesteps = 0
    # while num_timesteps < 5:
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)

        action_distribution, _ = actor_critic(state)
        if DETERMINISTIC:
            chosen_action = torch.argmax(action_distribution.probs).cpu().numpy()
        else:
            chosen_action = action_distribution.sample().cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(chosen_action)

        reward_sum += reward
        # reward_sum += reward - goal_distance(next_state["chars_crop"])

        # env.render()

        PIXEL_HISTORY.append(next_state["pixel"])

        if terminated or truncated:
            break

        state = process_observations(next_state)
        
        num_timesteps += 1

    env.close()

    print(reward_sum)

    fig = plt.figure()
    frame = plt.imshow(raw_state["pixel"])
    def update_animation_frame(i):
        frame.set_data(PIXEL_HISTORY[i])
        return [frame]
    animation = FuncAnimation(fig, update_animation_frame, frames=len(PIXEL_HISTORY), interval=500)
    
    plt.show()

    if args.record:
        animation.save(args.policy[:-4] + ".gif", dpi=300, writer=FFMpegWriter(fps=10))
        print("Video saved to path: " + args.policy[:-4] + ".gif")
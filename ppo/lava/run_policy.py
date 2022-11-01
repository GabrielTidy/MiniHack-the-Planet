
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
ENVIRONMENT_NAME = "MiniHack-LavaCross-Levitate-Potion-Inv-v0"
OBSERVATION_KEYS = ("chars", "message", "pixel")
ACTION_KEYS = (nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W,
        nethack.Command.QUAFF,
        nethack.Command.FIRE)
PENALTY_STEP = -0.01
PENALTY_TIME = -0.01
REWARD_WIN = 10
REWARD_LOSE = -10
DRINK_REWARD = 5
QUAFF_STATE = 10
QUAFF_REWARD = 0
LEVITATING_STATE = 10
THROW_PENALTY = -10

#whether to sample actions or always choose action with max probability
DETERMINISTIC = True

### End Hyperparameters ###

def positions(chars):
    player_pos = np.where(chars == 64)
    player_pos = [player_pos[0][0], player_pos[1][0]]
    goal_pos = np.where(chars == 62)
    goal_pos = [goal_pos[0][0], goal_pos[1][0]]
    return np.concatenate((player_pos, goal_pos))

def process_observations(obs, levitating=False):
    if levitating:
        return np.concatenate((np.array([0, LEVITATING_STATE]), positions(obs[OBSERVATION_KEYS[0]])))
    return np.concatenate((np.array([0, 0]), positions(obs[OBSERVATION_KEYS[0]])))

QUAFF_POTION_MESSAGE = np.array([87, 104, 97, 116, 32, 100, 111, 32, 121, 111, 117, 32, 119, 97, 110, 116, 32, 116, 111,
    32, 100, 114, 105, 110, 107, 63, 32, 91, 102, 32, 111, 114, 32, 63, 42, 93, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

DRINK_LEVITATION_POTION_MESSAGE = np.array([89, 111, 117, 32, 115, 116, 97, 114, 116, 32, 116, 111, 32, 102, 108, 111, 
    97, 116, 32, 105, 110, 32, 116, 104, 101, 32, 97, 105, 114, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p", "--policy", required=True, help="Policy checkpoint file to load")
    parser.add_argument("-r", "--record", default=False, help="If True, saves a video recording of the run (default=False")
    args = parser.parse_args()

    env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=OBSERVATION_KEYS, 
    reward_win=REWARD_WIN, reward_lose=REWARD_LOSE, penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP)

    env.seed(42)

    raw_state = env.reset()
    # goal_pos = get_goal_pos(raw_state[OBSERVATION_KEYS[0]])
    state = process_observations(raw_state)

    num_observations = state.shape[0]
    num_actions = env.action_space.n

    actor_critic = SeparateActorCritic(num_observations, num_actions)
    actor_critic.load_state_dict(torch.load(args.policy))

    PIXEL_HISTORY = [raw_state["pixel"]]

    env.render()

    reward_sum = 0.0
    num_timesteps = 0
    levitating = False
    # while num_timesteps < 5:
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)

        action_distribution, _ = actor_critic(state)
        if DETERMINISTIC:
            chosen_action = torch.argmax(action_distribution.probs).cpu().numpy()
        else:
            chosen_action = action_distribution.sample().cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(chosen_action)

        env.render()

        if np.array_equal(next_state["message"], DRINK_LEVITATION_POTION_MESSAGE):
            reward += DRINK_REWARD
            levitating = True
        elif (not np.array_equal(next_state["message"], QUAFF_POTION_MESSAGE)) and chosen_action == 5:
                reward += THROW_PENALTY
                terminated = True
        reward_sum += reward

        PIXEL_HISTORY.append(next_state["pixel"])

        if terminated or truncated:
            break

        state = process_observations(next_state, levitating)
        if np.array_equal(next_state["message"], QUAFF_POTION_MESSAGE):
            state[0] = QUAFF_STATE
        
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
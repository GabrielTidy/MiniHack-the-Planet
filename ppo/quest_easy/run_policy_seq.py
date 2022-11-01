
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
ENVIRONMENT_NAME = "MiniHack-Quest-Easy-v0"
OBSERVATION_KEYS = ("chars", "chars_crop", "pixel")
ACTION_KEYS = (nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W,
        nethack.Command.ZAP,
        nethack.Command.RUSH)
PENALTY_STEP = -0.1
PENALTY_TIME = 0.0
REWARD_WIN = 10
SOLIDIFY_LAVA_REWARD = 5
MONSTER_KILL_REWARD = 5
REWARD_LOSE = -25
GOAL_POS = (10, 51)

#whether to sample actions or always choose action with max probability
DETERMINISTIC = True

### End Hyperparameters ###

def positions(chars):
    player_pos = np.where(chars == 64)
    player_pos = [player_pos[0][0], player_pos[1][0]]
    goal_pos = [GOAL_POS[0], GOAL_POS[1]]
    return np.concatenate((player_pos, goal_pos))

def process_observations(obs):
    game_map = obs[OBSERVATION_KEYS[1]].flatten()
    return np.concatenate((positions(obs[OBSERVATION_KEYS[0]]), game_map))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p", "--policy", required=True, help="Policy checkpoint file to load")
    parser.add_argument("-r", "--record", default=False, help="If True, saves a video recording of the run (default=False")
    args = parser.parse_args()

    reward_manager = minihack.RewardManager()
    reward_manager.add_message_event(["The lava cools and solidifies."], reward=SOLIDIFY_LAVA_REWARD)
    reward_manager.add_kill_event("jackal", reward=MONSTER_KILL_REWARD)
    reward_manager.add_kill_event("lichen", reward=MONSTER_KILL_REWARD)
    reward_manager.add_coordinate_event(GOAL_POS, reward=REWARD_WIN)
    reward_manager.add_location_event("lava", reward=REWARD_LOSE, terminal_required=False)

    env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=OBSERVATION_KEYS, 
    penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP, reward_manager=reward_manager)

    env.seed(42)

    raw_state = env.reset()
    state = process_observations(raw_state)

    num_observations = state.shape[0]
    num_actions = 5

    actor_critic = SeparateActorCritic(num_observations, num_actions)
    actor_critic.load_state_dict(torch.load(args.policy))

    PIXEL_HISTORY = [raw_state["pixel"]]

    env.render()

    reward_sum = 0.0
    num_timesteps = 0
    next_action = -1
    # while num_timesteps < 100:
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_distribution, _ = actor_critic(state)

        if next_action != -1:#in ZAP sequence
            chosen_action = next_action
            if next_action == 5:#continue ZAP sequence
                next_action = 1
            elif next_action == 1:#end ZAP sequence
                next_action = -1
        else:#not in ZAP sequence - choose action from actor
            if DETERMINISTIC:
                chosen_action = torch.argmax(action_distribution.probs).cpu().numpy()
            else:
                chosen_action = action_distribution.sample().cpu().numpy()[0]
            if chosen_action == 4:#initiate ZAP sequence
                next_action = 5

        next_state, reward, terminated, truncated, _ = env.step(chosen_action)

        env.render()

        reward_sum += reward

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

from tensorboardX import SummaryWriter
import os
import numpy as np
import gym
import torch
from datetime import datetime
from actor_critic_network import SeparateActorCritic
import minihack
from nle import nethack
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


### Hyperparameters ###

#nethack minhack environment
ENVIRONMENT_NAME = "MiniHack-MazeWalk-15x15-v0"
OBSERVATION_KEYS = ("chars", "chars_crop")
TEST_OBSERVATION_KEYS = ("chars", "chars_crop", "pixel")
OBS_CROP = 3
ACTION_KEYS = (nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W)

#environment rewards
PENALTY_STEP = -1.0
PENALTY_TIME = 0.0
REWARD_WIN = 20
REWARD_LOSE = -10
NEW_POS_REWARD = 2.5

#character IDs of elements used in observation processing
PLAYER_ID = 64

#neural networks
LEARNING_RATE = 1e-4

#generalized advantage estimation
GAMMA = 0.99
GAE_LAMBDA = 0.95
NORMALIZE_ADVANTAGES = True

#ppo loss calculation
POLICY_CLIP = 0.2
CRITIC_LOSS_WEIGHT = 0.1
ENTROPY_LOSS_WEIGHT = 0.001

#training duration
BATCH_SIZE = 1024
MINI_BATCH_SIZE = 64
NUM_UPDATES_PER_PPO_EPOCH = 10
MAX_PPO_EPOCHS = 5000

#policy testing
TESTING_INTERVAL = 10
NUM_TESTS = 3
TARGET_REWARD = REWARD_WIN + 15

### End Hyperparameters ###


#environment observation processing
def process_observations(state, state_counts):
    cropped_chars = state[OBSERVATION_KEYS[0]][5:18, 32:45]

    player_pos = np.where(cropped_chars == PLAYER_ID)#the coordinates of the agent
    player_pos = [player_pos[0][0], player_pos[1][0]]

    player_view = state[OBSERVATION_KEYS[1]].flatten()#3x3 window around the agents current position

    #the number of times the agent has reached its current position, as well as the 4 positions directly above, below, right and left of the current position
    state_counts[player_pos[0]][player_pos[1]] += 1
    if player_pos[0] != 0:
        up = state_counts[player_pos[0] - 1][player_pos[1]]
    else:
        up = -1
    if player_pos[0] != 12:
        down = state_counts[player_pos[0] + 1][player_pos[1]]
    else:
        down = -1
    if player_pos[1] != 0:
        left = state_counts[player_pos[0]][player_pos[1] - 1]
    else:
        left = -1
    if player_pos[1] != 12:
        right = state_counts[player_pos[0]][player_pos[1] + 1]
    else:
        right = -1

    return np.concatenate((player_view, [up, left, state_counts[player_pos[0]][player_pos[1]], right, down])), state_counts

#when the agent reaches a brand new location in the maze, give a small positive reward for exploration
def reached_new_position(state, state_counts):
    cropped_chars = state[OBSERVATION_KEYS[0]][5:18, 32:45]
    player_pos = np.where(cropped_chars == PLAYER_ID)#the coordinates of the agent
    player_pos = [player_pos[0][0], player_pos[1][0]]

    return state_counts[player_pos[0]][player_pos[1]] == 0
    

#PPO functions
def generalized_advantage_estimate(state_values, rewards, terminal_masks):
    
    returns = []
    advantages = []
    advantage = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * state_values[step + 1] * terminal_masks[step] - state_values[step]
        advantage = delta + GAMMA * GAE_LAMBDA * terminal_masks[step] * advantage
        advantages.append(advantage)
        returns.append(advantage + state_values[step])
    
    advantages.reverse()
    returns.reverse()

    return advantages, returns

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

def make_minibatches(states, action_log_probs, actions, returns, advantages):
    batch_size = len(states)
    for _ in range(batch_size // MINI_BATCH_SIZE):
        random_indices = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[random_indices], action_log_probs[random_indices], actions[random_indices], returns[random_indices], advantages[random_indices]

def ppo_update(actor_critic, policy_optimizer, states, action_log_probs, actions, returns, advantages, time_step):

    returns_sum = 0.0
    advantages_sum = 0.0
    total_loss_sum = 0.0
    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    entropy_sum = 0.0

    batch_update_count = 0

    for _ in range(NUM_UPDATES_PER_PPO_EPOCH):
        
        for batch_states, batch_action_log_probs, batch_actions, batch_returns, batch_advantages in make_minibatches(states, action_log_probs, actions, returns, advantages):
            action_distributions, state_values = actor_critic(batch_states)
            entropy = action_distributions.entropy().mean()
            new_action_log_probs = action_distributions.log_prob(batch_actions)

            probability_ratio = (new_action_log_probs - batch_action_log_probs).exp()

            surrogate_loss_1 = probability_ratio * batch_advantages
            surrogate_loss_2 = torch.clamp(probability_ratio, 1.0 - POLICY_CLIP, 1.0 + POLICY_CLIP) * batch_advantages

            actor_loss = - torch.min(surrogate_loss_1, surrogate_loss_2).mean()
            critic_loss = (batch_returns - state_values).pow(2).mean()

            total_loss = CRITIC_LOSS_WEIGHT * critic_loss + actor_loss - ENTROPY_LOSS_WEIGHT * entropy

            policy_optimizer.zero_grad()
            total_loss.backward()
            policy_optimizer.step()

            returns_sum += batch_returns.mean()
            advantages_sum += batch_advantages.mean()
            total_loss_sum += total_loss
            actor_loss_sum += actor_loss
            critic_loss_sum += critic_loss
            entropy_sum += entropy
            
            batch_update_count += 1

    writer.add_scalar(ENVIRONMENT_NAME + "/returns", returns_sum / batch_update_count, time_step)
    writer.add_scalar(ENVIRONMENT_NAME + "/advantage", advantages_sum / batch_update_count, time_step)
    writer.add_scalar(ENVIRONMENT_NAME + "/total_loss", total_loss_sum / batch_update_count, time_step)
    writer.add_scalar(ENVIRONMENT_NAME + "/actor_loss", actor_loss_sum / batch_update_count, time_step)
    writer.add_scalar(ENVIRONMENT_NAME + "/critic_loss", critic_loss_sum / batch_update_count, time_step)
    writer.add_scalar(ENVIRONMENT_NAME + "/entropy", entropy_sum / batch_update_count, time_step)


#performance testing + video display/recording
def test_policy(env, actor_critic, savedir, test_num, deterministic=True):
    reward_sum = 0.0
    raw_state = env.reset()
    state_counts = np.zeros((13, 13))
    state, state_counts = process_observations(raw_state, state_counts)
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_distribution, _ = actor_critic(state)

        if deterministic:
            chosen_action = torch.argmax(action_distribution.probs).cpu().numpy()
        else:
            chosen_action = action_distribution.sample().cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(chosen_action)

        if (not (terminated or truncated)) and reached_new_position(next_state, state_counts):
            reward += NEW_POS_REWARD
        reward_sum += reward

        if terminated or truncated:
            break
        state, state_counts = process_observations(next_state, state_counts)
    
    return reward_sum

def record_policy(env, actor_critic, savedir, test_num, deterministic=True):
    reward_sum = 0.0
    raw_state = env.reset()
    PIXEL_HISTORY = [raw_state["pixel"][70:300, 500:730]]
    state_counts = np.zeros((13, 13))
    state, state_counts = process_observations(raw_state, state_counts)
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_distribution, _ = actor_critic(state)

        if deterministic:
            chosen_action = torch.argmax(action_distribution.probs).cpu().numpy()
        else:
            chosen_action = action_distribution.sample().cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(chosen_action)
        PIXEL_HISTORY.append(next_state["pixel"][70:300, 500:730])

        if (not (terminated or truncated)) and reached_new_position(next_state, state_counts):
            reward += NEW_POS_REWARD
        reward_sum += reward

        if terminated or truncated:
            break
        state, state_counts = process_observations(next_state, state_counts)
    
    # if reward_sum >= 15:
    print("Saving video...")
    fig = plt.figure()
    plt.title(ENVIRONMENT_NAME)
    plt.axis("off")
    frame = plt.imshow(PIXEL_HISTORY[0])
    def update_animation_frame(i):
        frame.set_data(PIXEL_HISTORY[i])
        return [frame]
    animation = FuncAnimation(fig, update_animation_frame, frames=len(PIXEL_HISTORY), interval=500)
    # plt.show()
    animation.save(savedir + "_%d.gif" % test_num, dpi=300, writer=PillowWriter(fps=10))
    print("Video saved to path: " + savedir + "_%d.gif" % test_num)
    plt.close(fig)
    
    return reward_sum


if __name__ == "__main__":

    #set up directories for storing data
    RUN_DIR = datetime.now().strftime("%b%d_%H-%M")
    if not os.path.exists(ENVIRONMENT_NAME):
        os.mkdir(ENVIRONMENT_NAME)
    if not os.path.exists(ENVIRONMENT_NAME + "/checkpoints"):
        os.mkdir(ENVIRONMENT_NAME + "/checkpoints")
    if not os.path.exists(ENVIRONMENT_NAME + "/videos"):
        os.mkdir(ENVIRONMENT_NAME + "/videos")
    if not os.path.exists(ENVIRONMENT_NAME + "/checkpoints/" + RUN_DIR):
        os.mkdir(ENVIRONMENT_NAME + "/checkpoints/" + RUN_DIR)
    if not os.path.exists(ENVIRONMENT_NAME + "/videos/" + RUN_DIR):
        os.mkdir(ENVIRONMENT_NAME + "/videos/" + RUN_DIR)
    writer = SummaryWriter(log_dir=ENVIRONMENT_NAME + "/checkpoints/" + RUN_DIR)

    device = "cpu"
    
    #initialize minihack environments
    training_env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=OBSERVATION_KEYS, 
        penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP, reward_win=REWARD_WIN, reward_lose=REWARD_LOSE, obs_crop_w=OBS_CROP, obs_crop_h=OBS_CROP)
    test_env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=TEST_OBSERVATION_KEYS, 
        penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP, reward_win=REWARD_WIN, reward_lose=REWARD_LOSE, obs_crop_w=OBS_CROP, obs_crop_h=OBS_CROP)
    training_env.seed(42)
    test_env.seed(42)

    state_counts = np.zeros((13, 13))

    #determine neural network input layer sizes based on the processing performed on the observations, and number of available actions
    state, state_counts = process_observations(training_env.reset(), state_counts)
    num_observations = state.shape[0]
    num_actions = training_env.action_space.n

    #initialize actor and critic neural networks
    actor_critic = SeparateActorCritic(num_observations, num_actions).to(device)
    policy_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    num_ppo_epochs = 0
    num_timesteps = 0
    best_reward = -float("inf")
    best_checkpoint = ""
    
    #training loop
    while True:

        state_history = []
        state_value_history = []
        action_log_probs_history = []
        action_history = []
        reward_history = []
        terminal_mask_history = []

        #generate trajectory batch
        for _ in range(BATCH_SIZE):
            state = torch.FloatTensor(state).to(device)
            action_distributions, state_value = actor_critic(state)

            chosen_action = action_distributions.sample()

            next_state, reward, terminated, truncated, _ = training_env.step(chosen_action.cpu().numpy())
            action_log_probs = action_distributions.log_prob(chosen_action)

            if (not (terminated or truncated)) and reached_new_position(next_state, state_counts):
                reward += NEW_POS_REWARD
            
            state_history.append(state)
            state_value_history.append(state_value)
            action_log_probs_history.append(action_log_probs)
            action_history.append(chosen_action)
            reward_history.append(reward)
            terminal_mask_history.append(1 - terminated)

            if terminated or truncated:
                state_counts = np.zeros((13, 13))
                state, state_counts = process_observations(training_env.reset(), state_counts)
            else:
                state, state_counts = process_observations(next_state, state_counts)

            num_timesteps += 1

        state_counts = np.zeros((13, 13))

        #gae calculation
        state = torch.FloatTensor(state).to(device)
        _, next_state_value = actor_critic(state)
        state_value_history = state_value_history + [next_state_value]
        advantage_history, return_history = generalized_advantage_estimate(state_value_history, reward_history, terminal_mask_history)

        #process trajectory data for PPO
        state_history = torch.stack((state_history))
        action_log_probs_history = torch.stack((action_log_probs_history)).detach()
        action_history = torch.stack((action_history))
        return_history = torch.stack((return_history)).detach()
        advantage_history = torch.stack((advantage_history)).detach()
        if NORMALIZE_ADVANTAGES:
            advantage_history = normalize(advantage_history)
        
        #perform PPO epoch
        ppo_update(actor_critic, policy_optimizer, state_history, action_log_probs_history, action_history, return_history, advantage_history, num_timesteps)
        num_ppo_epochs += 1

        #determine policy performance in test env
        if num_ppo_epochs % TESTING_INTERVAL == 0:
            test_rewards = [test_policy(test_env, actor_critic, ENVIRONMENT_NAME + "/videos/" + RUN_DIR + "/" + str(num_ppo_epochs), i) for i in range(NUM_TESTS)]
            average_reward = np.mean(test_rewards)

            print("After %d PPO epochs got reward of %.3f" % (num_ppo_epochs, average_reward))
            writer.add_scalar(ENVIRONMENT_NAME + "/test-reward", average_reward, num_ppo_epochs)

            if best_reward < average_reward:
                best_reward = average_reward
                best_checkpoint = "%d_%d" % (num_ppo_epochs, best_reward)
                torch.save(actor_critic.state_dict(), ENVIRONMENT_NAME + "/checkpoints/" + RUN_DIR + "/" + best_checkpoint + ".dat")
                print("New best policy saved.")
                
            if average_reward >= TARGET_REWARD:
                print("Target reward reached!")
                record_policy(test_env, actor_critic, ENVIRONMENT_NAME + "/videos/" + RUN_DIR + "/" + str(num_ppo_epochs), 0)
                break
            elif num_ppo_epochs >= MAX_PPO_EPOCHS:
                print("Max number of PPO epochs reached.")
                break

    #cleanup
    training_env.close()
    test_env.close()
    writer.close()

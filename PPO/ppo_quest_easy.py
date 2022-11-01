
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
ENVIRONMENT_NAME = "MiniHack-Quest-Easy-v0"
OBSERVATION_KEYS = ("chars", "chars_crop", "message")
TEST_OBSERVATION_KEYS = ("chars", "chars_crop", "message", "pixel")
OBS_CROP = 3
ACTION_KEYS = (nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W,
        nethack.Command.ZAP,
        nethack.Command.RUSH)
ZAP_ACTION = 4
RUSH_ACTION = 5
EAST_ACTION = 1

#environment rewards
PENALTY_STEP = -0.1
PENALTY_TIME = -0.01
REWARD_WIN = 15
SOLIDIFY_LAVA_REWARD = 3
MONSTER_KILL_REWARD = 5
MISS_ZAP_REWARD = -0.1
REWARD_LOSE = -25

#observation processing (additional state values and character IDs)
GOAL_POS = (10, 51)
PLAYER_ID = 64
JACKAL_ID = 100
LICHEN_ID = 70

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
TARGET_REWARD = REWARD_WIN + SOLIDIFY_LAVA_REWARD + MONSTER_KILL_REWARD*2 - 5

### End Hyperparameters ###


#environment observation processing
def positions(chars):
    player_pos = np.where(chars == PLAYER_ID)
    player_pos = [player_pos[0][0], player_pos[1][0]]

    goal_pos = [GOAL_POS[0], GOAL_POS[1]]

    num_monsters = 0
    jackal_pos = np.where(chars == JACKAL_ID)
    if jackal_pos[0].shape[0] == 0:
        jackal_pos = [0, 0]
    else:
        num_monsters += 1
        jackal_pos = [jackal_pos[0][0], jackal_pos[1][0]]
    
    lichen_pos = np.where(chars == LICHEN_ID)
    if lichen_pos[0].shape[0] == 0:
        lichen_pos = [0, 0]
    else:
        num_monsters += 1
        lichen_pos = [lichen_pos[0][0], lichen_pos[1][0]]

    return np.concatenate((player_pos, goal_pos, [num_monsters], jackal_pos, lichen_pos))

def process_observations(obs):
    return np.concatenate((positions(obs[OBSERVATION_KEYS[0]]), obs[OBSERVATION_KEYS[1]].flatten()))
    

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
def test_policy(env, actor_critic, savedir, deterministic=True):
    reward_sum = 0.0
    raw_state = env.reset()
    PIXEL_HISTORY = [raw_state["pixel"][90:240, 375:860]]
    state = process_observations(raw_state)
    next_action = -1
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_distribution, _ = actor_critic(state)

        if next_action != -1:#in ZAP sequence
            chosen_action = next_action
            if next_action == RUSH_ACTION:#continue ZAP sequence
                next_action = EAST_ACTION
            elif next_action == EAST_ACTION:#end ZAP sequence
                next_action = -1
        else:#not in ZAP sequence - choose action from actor
            if deterministic:
                chosen_action = torch.argmax(action_distribution.probs).cpu().numpy()
            else:
                chosen_action = action_distribution.sample().cpu().numpy()
            if chosen_action == ZAP_ACTION:#initiate ZAP sequence
                next_action = RUSH_ACTION

        next_state, reward, terminated, truncated, _ = env.step(chosen_action)
        PIXEL_HISTORY.append(next_state["pixel"][90:240, 375:860])
        reward_sum += reward

        if terminated or truncated:
            break
        state = process_observations(next_state)
    
    if reward_sum >= TARGET_REWARD:
        print("Targer reward reached! Saving video...")
        fig = plt.figure()
        plt.title(ENVIRONMENT_NAME)
        plt.axis("off")
        frame = plt.imshow(PIXEL_HISTORY[0])
        def update_animation_frame(i):
            frame.set_data(PIXEL_HISTORY[i])
            return [frame]
        animation = FuncAnimation(fig, update_animation_frame, frames=len(PIXEL_HISTORY), interval=500)
        plt.show()
        animation.save(savedir + ".gif", dpi=300, writer=PillowWriter(fps=10))
        print("Video saved to path: " + savedir + ".gif")
    
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #initialize minihack environments
    reward_manager = minihack.RewardManager()
    reward_manager.add_message_event(["The lava cools and solidifies."], reward=SOLIDIFY_LAVA_REWARD)
    reward_manager.add_message_event(["Nothing happens."], reward=MISS_ZAP_REWARD)
    reward_manager.add_kill_event("jackal", reward=MONSTER_KILL_REWARD)
    reward_manager.add_kill_event("lichen", reward=MONSTER_KILL_REWARD)
    reward_manager.add_coordinate_event(GOAL_POS, reward=REWARD_WIN)
    reward_manager.add_location_event("lava", reward=REWARD_LOSE, terminal_required=False)
    training_env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=OBSERVATION_KEYS, 
        penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP, obs_crop_w=OBS_CROP, obs_crop_h=OBS_CROP, reward_manager=reward_manager)
    test_env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=TEST_OBSERVATION_KEYS, 
        penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP, obs_crop_w=OBS_CROP, obs_crop_h=OBS_CROP, reward_manager=reward_manager)
    training_env.seed(42)
    test_env.seed(42)

    #determine neural network input layer sizes based on the processing performed on the observations, and number of available actions
    state = process_observations(training_env.reset())
    num_observations = state.shape[0]
    num_actions = training_env.action_space.n - 1#subtract 1 as the extra action is only used in the ZAP sequence
    # num_actions = training_env.action_space.n

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
        next_action = -1

        #generate trajectory batch
        for _ in range(BATCH_SIZE):
            state = torch.FloatTensor(state).to(device)
            action_distributions, state_value = actor_critic(state)

            if next_action != -1:#in ZAP sequence
                log_action = torch.tensor(2, dtype=torch.int8)
                chosen_action = torch.tensor(next_action, dtype=torch.int8)
                if next_action == RUSH_ACTION:#continue ZAP sequence
                    next_action = EAST_ACTION
                elif next_action == EAST_ACTION:#end ZAP sequence
                    next_action = -1
            else:#not in ZAP sequence - choose action from actor
                chosen_action = action_distributions.sample()
                log_action = chosen_action
                if chosen_action.cpu().numpy() == ZAP_ACTION:#initiate ZAP sequence
                    next_action = RUSH_ACTION

            next_state, reward, terminated, truncated, _ = training_env.step(chosen_action.cpu().numpy())
            action_log_probs = action_distributions.log_prob(log_action)
            
            state_history.append(state)
            state_value_history.append(state_value)
            action_log_probs_history.append(action_log_probs)
            action_history.append(log_action)
            reward_history.append(reward)
            terminal_mask_history.append(1 - terminated)

            if terminated or truncated:
                state = process_observations(training_env.reset())
                next_action = -1#end ZAP sequence
            else:
                state = process_observations(next_state)

            num_timesteps += 1

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
            test_reward = test_policy(test_env, actor_critic, ENVIRONMENT_NAME + "/videos/" + RUN_DIR + "/" + str(num_ppo_epochs))

            print("After %d PPO epochs got reward of %.3f" % (num_ppo_epochs, test_reward))
            writer.add_scalar(ENVIRONMENT_NAME + "/test-reward", test_reward, num_ppo_epochs)

            if best_reward < test_reward:
                best_reward = test_reward
                best_checkpoint = "%d_%d" % (num_ppo_epochs, best_reward)
                torch.save(actor_critic.state_dict(), ENVIRONMENT_NAME + "/checkpoints/" + RUN_DIR + "/" + best_checkpoint + ".dat")
                print("New best policy saved.")
                
            if test_reward >= TARGET_REWARD:
                break
            elif num_ppo_epochs >= MAX_PPO_EPOCHS:
                print("Max number of PPO epochs reached.")
                break

    #cleanup
    training_env.close()
    test_env.close()
    writer.close()

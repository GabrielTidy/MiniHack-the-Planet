
from tensorboardX import SummaryWriter
import os
import numpy as np
import gym
import torch
from datetime import datetime
from actor_critic_network import SeparateActorCritic
import minihack
from nle import nethack


### Hyperparameters ###

#nethack minhack environment
ENVIRONMENT_NAME = "MiniHack-Quest-Easy-v0"
OBSERVATION_KEYS = ("chars", "message")
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
MAX_PPO_EPOCHS = 1000

#policy testing
TESTING_INTERVAL = 10
NUM_TESTS = 1
TARGET_REWARD = REWARD_WIN + SOLIDIFY_LAVA_REWARD + MONSTER_KILL_REWARD*2

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
    return positions(obs[OBSERVATION_KEYS[0]])
    

#PPO functions
def generalized_advantage_estimate(state_values, rewards, terminal_masks):
    
    returns = []
    advantages = []
    advantage = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * state_values[step + 1] * terminal_masks[step] - state_values[step]
        advantage = delta + GAMMA * GAE_LAMBDA * terminal_masks[step] * advantage
        # advantages.insert(0, advantage)
        # returns.insert(0, advantage + state_values[step])
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
def test_policy(env, actor_critic, deterministic=True):
    rewards_sums = 0.0
    raw_state = env.reset()
    state = process_observations(raw_state)
    next_action = -1
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
            if deterministic:
                chosen_action = torch.argmax(action_distribution.probs).cpu().numpy()
            else:
                chosen_action = action_distribution.sample().cpu().numpy()
            if chosen_action == 4:#initiate ZAP sequence
                next_action = 5

        next_state, reward, terminated, truncated, _ = env.step(chosen_action)
        
        rewards_sums += reward

        if terminated or truncated:
            return rewards_sums

        state = process_observations(next_state)


if __name__ == "__main__":

    RUN_DIR = datetime.now().strftime("%b%d_%H-%M")
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists("checkpoints/" + RUN_DIR):
        os.mkdir("checkpoints/" + RUN_DIR)

    writer = SummaryWriter(log_dir="checkpoints/" + RUN_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    reward_manager = minihack.RewardManager()
    reward_manager.add_message_event(["The lava cools and solidifies."], reward=SOLIDIFY_LAVA_REWARD)
    reward_manager.add_kill_event("jackal", reward=MONSTER_KILL_REWARD)
    reward_manager.add_kill_event("lichen", reward=MONSTER_KILL_REWARD)
    reward_manager.add_coordinate_event(GOAL_POS, reward=REWARD_WIN)
    reward_manager.add_location_event("lava", reward=REWARD_LOSE, terminal_required=False)
    
    training_env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=OBSERVATION_KEYS, 
        penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP, reward_manager=reward_manager)
    test_env = gym.make(ENVIRONMENT_NAME, new_step_api=True, actions=ACTION_KEYS, observation_keys=OBSERVATION_KEYS, 
        penalty_time=PENALTY_TIME, penalty_step=PENALTY_STEP, reward_manager=reward_manager)
    
    training_env.seed(42)
    test_env.seed(42)

    raw_state = training_env.reset()
    state = process_observations(raw_state)

    num_observations = state.shape[0]
    num_actions = 5

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
                log_action = torch.tensor(4, dtype=torch.int8)
                chosen_action = torch.tensor(next_action, dtype=torch.int8)
                if next_action == 5:#continue ZAP sequence
                    next_action = 1
                elif next_action == 1:#end ZAP sequence
                    next_action = -1
            else:#not in ZAP sequence - choose action from actor
                chosen_action = action_distributions.sample()
                log_action = chosen_action
                if chosen_action.cpu().numpy() == 4:#initiate ZAP sequence
                    next_action = 5

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
            test_rewards = np.array([test_policy(test_env, actor_critic) for _ in range(NUM_TESTS)])
            average_reward = np.mean(test_rewards)

            print("After %d PPO epochs got reward of %.3f" % (num_ppo_epochs, average_reward))
            writer.add_scalar(ENVIRONMENT_NAME + "/test-reward", average_reward, num_ppo_epochs)

            torch.save(actor_critic.state_dict(), "checkpoints/" + RUN_DIR + "/" + "%d_%d" % (num_ppo_epochs, average_reward) + ".dat")

            if best_reward < average_reward:
                best_reward = average_reward
                best_checkpoint = "%d_%d" % (num_ppo_epochs, best_reward)
                print("New best policy saved.")
                
            if average_reward >= TARGET_REWARD:
                print("Target reward reached!")
                break
            elif num_ppo_epochs >= MAX_PPO_EPOCHS:
                print("Max number of PPO epochs reached.")
                break

    training_env.close()
    test_env.close()
    writer.close()

from minihack import RewardManager
import minihack
from nle import nethack
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from agent import DQNAgent
from replay_buffer import ReplayBuffer

def train_agent(env,agent,save_agent=False, env_name = "", agent_name="default", write_stats=False):
    global hyper_params
    
    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    losses=[]

    obs_type = 'chars_crop'
    state = env.reset()[obs_type]

    # lava_count = 0
    # key_count = 0
    # last_action = None
    # levitating = False
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()

        if sample<eps_threshold:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        n_state,reward,done,_ = env.step(action)
        n_state = n_state[obs_type]

    #     if levitating and action in np.arange(8):
    #         reward += 0.5
    #     if action ==8:
    #         key_count += 1
    #         if key_count == 1:
    #             print('drank potion')
    #             reward += 5
    #     if(last_action ==8 and action == 9):
    #         lava_count += 1
    #         if lava_count == 1:
    #             print('used potion')
    #             reward = 5
    #             levitating = True
    #     elif (last_action !=8 and action == 9):
    #             reward -= 10
    #             done = True


        agent.r_buffer.add(state,action,reward,n_state,float(done))
        state = n_state

        last_action = action


        episode_rewards[-1] += reward
        if done:
            state = env.reset()[obs_type]
#             key_count =0
#             lava_count=0
#             last_action =None
#             levitating = False
            episode_rewards.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            losses.append(agent.optimise_td_loss())

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
    
    if save_agent:
        agent.save_model(agent_name)
        
    if write_to_file:
        np.savetxt(f"eps_rewards_{env_name}.csv", 
           np.reshape(episode_rewards,(-1,1)),
           header = 'Episodic rewards',
           fmt ='%10.5f')
        np.savetxt(f"step_losses_{env_name}.csv", 
           np.reshape(losses,(-1,1)),
           header = 'loss',
           fmt ='%10.5f')
           
def test_agent(test_env,agent,record=False,fname='recent'):
    obs_type = 'chars_crop'
    
    raw_state = test_env.reset()
    state = raw_state[obs_type]

    PIXEL_HISTORY = [raw_state["pixel"]]
    episode_rewards = 0.0

    while True:

        action = agent.act(state)
        next_state, reward, done, _ = test_env.step(action)

        episode_rewards+=reward
        PIXEL_HISTORY.append(next_state["pixel"])

        if done:
            break

        state = next_state[obs_type]

    test_env.close()

    if record:
        fig = plt.figure()
        anim = plt.imshow(raw_state["pixel"])
        def update_animation_frame(i):
            anim.set_data(PIXEL_HISTORY[i])
            return [anim]
        ani = FuncAnimation(fig, update_animation_frame, frames=len(PIXEL_HISTORY), interval=500)
        ani.save(f"{fname}.mp4", dpi=300, writer=FFMpegWriter(fps=1))

        print("Video saved")
        
    return episode_rewards 
    
if __name__== '__main__':

	hyper_params = {
	    "seed": 42,  # which seed to use
	    "room-env": "MiniHack-Room-Random-15x15-v0",  # for room task
	    "maze-env": "MiniHack-MazeWalk-15x15-v0",  # for maze task
	    "lava-env": "MiniHack-LavaCross-Levitate-Potion-Inv-v0",  # for lava crossing task
	    "questE-env": "MiniHack-Quest-Easy-v0",  # name of the game
	    "questH-env": "MiniHack-Quest-Hard-v0",
	    "replay-buffer-size": int(5e3),  # replay buffer size
	    "learning-rate": 1e-4,  # learning rate for Adam optimizer
	    "discount-factor": 0.99,  # discount factor
	    "num-steps": int(1e6),  # total number of steps to run the environment for
	    "batch-size": 256,  # number of transitions to optimize at the same time
	    "learning-starts": 10000,  # number of steps before learning starts
	    "learning-freq": 3,  # number of iterations between every optimization step
	    "use-double-dqn": True,  # use double deep Q-learning
	    "target-update-freq": 1000,  # number of iterations between every target network update
	    "eps-start": 1.0,  # e-greedy start threshold
	    "eps-end": 0.01,  # e-greedy end threshold
	    "eps-fraction": 0.4,  # fraction of num-steps
	    "print-freq": 10,
	}

	np.random.seed(hyper_params["seed"])
	random.seed(hyper_params["seed"])

	NAV_ACTIONS = tuple(nethack.CompassDirection)
	SKILL_ACTIONS = {'lava-env':(nethack.Command.QUAFF,nethack.Command.FIRE),
		        'quest-env': (nethack.Command.PICKUP,
		                        nethack.Command.APPLY,
		                        nethack.Command.ZAP,
		                        nethack.Command.WEAR,
		                        nethack.Command.PUTON,
		                        nethack.Command.QUAFF,
		                        nethack.Command.FIRE),                
		        }

	# reward_manager = RewardManager()
	# reward_manager.add_kill_event("minotaur", reward=1, terminal_required=False)
	# strings = list()
	# strings.append("The door opens.")
	# reward_manager.add_message_event(strings, reward=1, terminal_required=True)
	# strings = list()
	# strings.append("It's solid stone.")
	# reward_manager.add_message_event("It's solid stone.", reward=-0.5, terminal_required=False, repeatable=True)

	# reward_manager.add_event(IdleEvent(0.5, True, True, False))
	
	rand_nums = np.random.randint(1000,size=3)
	observations = ['chars_crop','message']
	env_name = 'maze' #options: room/lava/maze/questE/questH
	env = gym.make(hyper_params[f'{env_name}-env'],
                observation_keys = observations,
#                 obs_crop_w=15,
#                 penalty_time=-0.01,
#                 penalty_step=-0.1,
                reward_lose=-1,
                reward_win=1,
                seeds = rand_nums,
#                 reward_manager=reward_manager,
                actions = NAV_ACTIONS)

	replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

	agent = DQNAgent(np.prod(env.observation_space[obs_type].shape), env.action_space.n,replay_buffer, hyper_params["use-double-dqn"],hyper_params["learning-rate"],hyper_params["batch-size"],hyper_params["discount-factor"])

	train_agent(env,agent,save_agent=True,env_name =env_name, write_stats=True)
	
	test_env = gym.make(hyper_params[f'{env_name}-env'],
                    observation_keys = observations +['pixel'],
                    reward_lose=-1,
                    reward_win=1,
                    seeds = [42],
    #                 reward_manager=reward_manager,
                    actions = ACTIONS)
	agent.load_model('default')
	
	_ =test_agent(test_env,agent,record=True)


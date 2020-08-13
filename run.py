import torch
from agent import IQN_Agent
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
import math
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import time
import gym
import argparse
import wrapper


def evaluate(eps, frame, eval_runs=5):
    """
    Makes an evaluation run with the current epsilon
    """
    if "-ram" in env_name or env_name == "CartPole-v0" or env_name == "LunarLander-v2": 
        env = gym.make(env_name)
    else:
        env = wrapper.make_env(env_name)
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        while True:
            action = agent.act(state, eps)
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
        
    writer.add_scalar("Reward", np.mean(reward_batch), frame)



def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01, eval_every=1000, eval_runs=5):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    output_history = []
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    i_episode = 1
    state = env.reset()
    score = 0                  
    for frame in range(1, frames+1):

        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done, writer)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)

        # evaluation runs
        if frame % eval_every == 0:
            evaluate(eps, frame, eval_runs)
        
        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), frame)
            output_history.append(np.mean(scores_window))
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
            i_episode +=1 
            state = env.reset()
            score = 0              

    return output_history


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-agent", type=str, choices=["iqn",
                                                     "iqn+per",
                                                     "noisy_iqn",
                                                     "noisy_iqn+per",
                                                     "dueling",
                                                     "dueling+per", 
                                                     "noisy_dueling",
                                                     "noisy_dueling+per"
                                                     ], default="iqn", help="Specify which type of IQN agent you want to train, default is IQN - baseline!")
    
    parser.add_argument("-env", type=str, default="CartPole-v0", help="Name of the Environment, default = CartPole-v0")
    parser.add_argument("-frames", type=int, default=60000, help="Number of frames to train, default = 60000")
    parser.add_argument("-eval_every", type=int, default=1000, help="Evaluate every x frames, default = 1000")
    parser.add_argument("-eval_runs", type=int, default=5, help="Number of evaluation runs, default = 5")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="Batch size for updating the DQN, default = 8")
    parser.add_argument("-layer_size", type=int, default=512, help="Size of the hidden layer, default=512")
    parser.add_argument("-n_step", type=int, default=1, help="Multistep IQN, default = 1")
    parser.add_argument("-m", "--memory_size", type=int, default=int(1e5), help="Replay memory size, default = 1e5")
    parser.add_argument("-u", "--update_every", type=int, default=1, help="Update the network every x steps, default = 1")
    parser.add_argument("-lr", type=float, default=5e-4, help="Learning rate, default = 5e-4")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Soft update parameter tat, default = 1e-2")
    parser.add_argument("-eps_frames", type=int, default=5000, help="Linear annealed frames for Epsilon, default = 5000")
    parser.add_argument("-min_eps", type=float, default = 0.025, help="Final epsilon greedy value, default = 0.025")
    parser.add_argument("-info", type=str, help="Name of the training run")
    parser.add_argument("-save_model", type=int, choices=[0,1], default=0, help="Specify if the trained network shall be saved or not, default is 0 - not saved!")

    args = parser.parse_args()
    writer = SummaryWriter("runs/"+args.info)
    seed = args.seed
    BUFFER_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.lr
    UPDATE_EVERY = args.update_every
    n_step = args.n_step
    env_name = args.env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    np.random.seed(seed)
    if "-ram" in args.env or args.env == "CartPole-v0" or args.env == "LunarLander-v2": 
        env = gym.make(args.env)
    else:
        env = wrapper.make_env(args.env)

    env.seed(seed)
    action_size = env.action_space.n
    state_size = env.observation_space.shape

    agent = IQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        network=args.agent,
                        layer_size=args.layer_size,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY, 
                        device=device, 
                        seed=seed)



    # set epsilon frames to 0 so no epsilon exploration
    if "noisy" in args.agent:
        eps_fixed = True
    else:
        eps_fixed = False

    t0 = time.time()
    final_average100 = run(frames = args.frames, eps_fixed=eps_fixed, eps_frames=args.eps_frames, min_eps=args.min_eps, eval_every=args.eval_every, eval_runs=args.eval_runs)
    t1 = time.time()
    
    print("Training time: {}min".format(round((t1-t0)/60,2)))
    if args.save_model:
        torch.save(agent.qnetwork_local.state_dict(), args.info+".pth")

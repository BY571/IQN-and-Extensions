import torch
from agent import IQN_Agent
import numpy as np
import random

import wandb
from collections import deque
import time
import gym
import argparse
import wrapper
import MultiPro
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def evaluate(eps, frame, eval_runs=5):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0), 0.001, eval=True)
            state, reward, done, _ = env.step(action[0].item())
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)



def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01, eval_every=1000, eval_runs=5, worker=1):
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
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    i_episode = 1
    state = env.reset()
    score = 0                  
    for frame in range(1, frames+1):
        action = agent.act(state[None, :, :, :], eps)
        next_state, reward, done, _ = env.step(action[0]) #returns np.stack(obs), np.stack(action) ...
        #for s, a, r, ns, d in zip(state, action, reward, next_state, done):
        reward = 0
        agent.step(state, action, reward, next_state, done, wandb)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value (until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            #if frame < eps_frames:
            eps = max(eps_start - ((frame*d_eps)/eps_frames), min_eps)
            #else:
            #   eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)
        
        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            wandb.log({"Average100": np.mean(scores_window)}, step=frame*worker)
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)))
            i_episode +=1 
            # evaluation runs
            if i_episode % eval_every == 0:
                eval_rewards = evaluate(eps, frame*worker, eval_runs) 
                wandb.log({"eval_reward": eval_rewards}, step=frame*worker)
            state = env.reset()
            score = 0
   




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
    
    parser.add_argument("-env", type=str, default="SuperMarioBros-v0", help="Name of the Environment, default = BreakoutNoFrameskip-v4")
    parser.add_argument("-frames", type=int, default=1_000_000, help="Number of frames to train, default = 10 mio")
    parser.add_argument("-eval_every", type=int, default=5, help="Evaluate every x epochs")
    parser.add_argument("-eval_runs", type=int, default=2, help="Number of evaluation runs, default = 2")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("-N", type=int, default=8, help="Number of Quantiles, default = 8")
    parser.add_argument("-munchausen", type=int, default=0, choices=[0,1], help="Use Munchausen RL loss for training if set to 1 (True), default = 0")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size for updating the DQN, default = 32")
    parser.add_argument("-layer_size", type=int, default=512, help="Size of the hidden layer, default=512")
    parser.add_argument("-n_step", type=int, default=1, help="Multistep IQN, default = 1")
    parser.add_argument("-m", "--memory_size", type=int, default=int(1e5), help="Replay memory size, default = 1e5")
    parser.add_argument("-lr", type=float, default=0.00025, help="Learning rate, default = 2.5e-4")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Soft update parameter tau, default = 1e-3")
    parser.add_argument("-eps_frames", type=int, default=1000000, help="Linear annealed frames for Epsilon, default = 1mio")
    parser.add_argument("-min_eps", type=float, default = 0.01, help="Final epsilon greedy value, default = 0.01")
    parser.add_argument("-info", type=str, help="Name of the training run")
    parser.add_argument("-save_model", type=int, choices=[0,1], default=1, help="Specify if the trained network shall be saved or not, default is 1 - save model!")
    parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel Environments. Batch size increases proportional to number of worker. not recommended to have more than 4 worker, default = 1")

    args = parser.parse_args()

    seed = args.seed
    BUFFER_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.lr
    n_step = args.n_step
    env_name = args.env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if "-ram" in args.env or args.env == "CartPole-v0" or args.env == "LunarLander-v2": 
    #     envs = MultiPro.SubprocVecEnv([lambda: gym.make(args.env) for i in range(args.worker)])
    #     eval_env = gym.make(args.env)
    # else:
    #     envs = MultiPro.SubprocVecEnv([lambda: wrapper.make_env(args.env) for i in range(args.worker)])
    #     eval_env = wrapper.make_env(args.env)
    if "SuperMario" in args.env:
        env = gym_super_mario_bros.make(args.env)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = wrapper.make_env(env)
    else:
        env = gym.make(args.env)
        env = wrapper.make_env(env)
    
    env.seed(seed)

    action_size = env.action_space.n
    state_size = env.observation_space.shape

    agent = IQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        embedding_size=1024,
                        network=args.agent,
                        munchausen=args.munchausen,
                        layer_size=args.layer_size,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA,  
                        N=args.N,
                        worker=args.worker,
                        device=device)



    # set epsilon frames to 0 so no epsilon exploration
    if "noisy" in args.agent:
        eps_fixed = True
    else:
        eps_fixed = False

    with wandb.init(project="IQN", name="iqn", config=args):
        run(frames = args.frames//args.worker, eps_fixed=eps_fixed, eps_frames=args.eps_frames//args.worker, min_eps=args.min_eps, eval_every=args.eval_every//args.worker, eval_runs=args.eval_runs, worker=args.worker)

    if args.save_model:
        torch.save(agent.qnetwork_local.state_dict(), args.info+".pth")

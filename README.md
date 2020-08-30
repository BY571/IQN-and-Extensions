# Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning and Extensions
PyTorch Implementation of Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning with additional extensions like PER, Noisy layer and N-step bootstrapping. Creating a new Rainbow-DQN version.


### Implementations
- Baseline IQN [Notebook](https://github.com/BY571/IQN/blob/master/IQN-DQN.ipynb)
- Script Version with all extensions: [IQN](https://github.com/BY571/IQN/blob/master/run.py)

### Extensions
- Dueling IQN
- Noisy layer
- N-step bootstrapping 
- [Munchausen](https://medium.com/analytics-vidhya/munchausen-reinforcement-learning-9876efc829de) RL 

## Train
With the script version it is possible to train on simple environments like CartPole-v0 and LunarLander-v2 or on Atari games with image inputs!

To run the script version:
`python run.py -info iqn_run1`

To run the script version on the Atari game Pong:
`python run.py -env PongNoFrameskip-v4 -info iqn_pong1`

#### Other hyperparameter and possible inputs
To see the options:
`python run.py -h`

    -agent, choices=["iqn","iqn+per","noisy_iqn","noisy_iqn+per","dueling","dueling+per", "noisy_dueling","noisy_dueling+per"], Specify which type of IQN agent you want to train, default is IQN - baseline!
    -env,  Name of the Environment, default = CartPole-v0
    -frames, Number of frames to train, default = 60000
    -eval_every, Evaluate every x frames, default = 1000
    -eval_runs, Number of evaluation runs, default = 5
    -seed, Random seed to replicate training runs, default = 1
    -munchausen, choices=[0,1], Use Munchausen RL loss for training if set to 1 (True), default = 0
    -bs, --batch_size, Batch size for updating the DQN, default = 8
    -layer_size, Size of the hidden layer, default=512
    -n_step, Multistep IQN, default = 1
    -N, Number of quantiles, default = 8
    -m, --memory_size, Replay memory size, default = 1e5
    -u, --update_every, Update the network every x steps, default = 1
    -lr, Learning rate, default = 5e-4
    -g, --gamma, Discount factor gamma, default = 0.99
    -t, --tau, Soft update parameter tat, default = 1e-2
    -eps_frames, Linear annealed frames for Epsilon, default = 5000
    -min_eps, Final epsilon greedy value, default = 0.025
    -info, Name of the training run
    -save_model, choices=[0,1]  Specify if the trained network shall be saved or not, default is 0 - not saved!

### Observe training results
  `tensorboard --logdir=runs`
  

#### Dependencies
Trained and tested on:
<pre>
Python 3.5.6 
PyTorch 1.4.0  
Numpy 1.15.2 
gym 0.10.11 
</pre>

## CartPole Results
IQN and Extensions (default hyperparameter):
![alttext](/imgs/IQN_CP_.png)

Dueling IQN and Extensions (default hyperparameter):
![alttext](/imgs/Dueling_IQN_CP_.png)


## Atari Results
IQN and M-IQN comparison (only trained for 500000 frames ~ 140 min).


**Hyperparameter:**
- frames 500000
- eps_frames 75000
- min_eps 0.025
- eval_every 10000 
- lr 1e-4 
- t 5e-3 
- m 15000 
- N 32

![alttext](/imgs/IQN_MIQN_BREAKOUT_.png)

## ToDo:
- Comparison plot for n-step bootstrapping (n-step bootstrapping with n=3 seems to give a strong boost in learning compared to one step bootstrapping, plots will follow) 
- Performance plot for Pong compared with Rainbow
- adding [Munchausen](https://medium.com/analytics-vidhya/munchausen-reinforcement-learning-9876efc829de) RL &#x2611;


## Help and issues:
Im open for feedback, found bugs, improvements or anything. Just leave me a message or contact me.

### Paper references:

- [IQN](https://arxiv.org/abs/1806.06923)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Noisy layer](https://arxiv.org/pdf/1706.10295.pdf)
- [C51](https://arxiv.org/pdf/1707.06887.pdf)
- [PER](https://arxiv.org/pdf/1511.05952.pdf)


## Author
- Sebastian Dittert

**Feel free to use this code for your own projects or research.**
For citation:
```
@misc{IQN and Extensions,
  author = {Dittert, Sebastian},
  title = {Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning and Extensions},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BY571/IQN}},
}

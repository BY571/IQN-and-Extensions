import torch
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import random
import math
from ReplayBuffers import ReplayBuffer, PrioritizedReplay
from model import IQN

class IQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 network,
                 munchausen,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 N,
                 worker,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.network = network
        self.munchausen = munchausen
        self.seed = random.seed(seed)
        self.seed_t = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.N = N
        self.K = 32
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        self.GAMMA = GAMMA
        
        self.BATCH_SIZE = BATCH_SIZE * worker
        self.Q_updates = 0
        self.n_step = n_step
        self.worker = worker
        self.UPDATE_EVERY = worker
        self.last_action = None

        if "noisy" in self.network:
            noisy = True
        else:
            noisy = False
        
        if "duel" in self.network:
            duel = True
        else:
            duel = False

        
        # IQN-Network
        self.qnetwork_local = IQN(state_size, action_size,layer_size, n_step, seed, N, dueling=duel, noisy=noisy, device=device).to(device)
        self.qnetwork_target = IQN(state_size, action_size,layer_size, n_step, seed,N, dueling=duel, noisy=noisy, device=device).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)
        
        # Replay memory
        if "per" in self.network:
            self.per = 1
            self.memory = PrioritizedReplay(BUFFER_SIZE, self.BATCH_SIZE, seed=seed, gamma=self.GAMMA, n_step=n_step, parallel_env=worker)
        else:
            self.per = 0
            self.memory = ReplayBuffer(BUFFER_SIZE, self.BATCH_SIZE, self.device, seed, self.GAMMA, n_step, worker)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, writer):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                if not self.per:
                    loss = self.learn(experiences)
                else:
                    loss = self.learn_per(experiences)
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0., eval=False):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """


        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            if len(self.state_size) > 1:
                state = torch.from_numpy(state).float().to(self.device)#.expand(self.K, self.state_size[0], self.state_size[1],self.state_size[2])        
            else:
                state = torch.from_numpy(state).float().to(self.device)#.expand(self.K, self.state_size[0])
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(state)#.mean(0)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action
        else:
            if eval:
                action = random.choices(np.arange(self.action_size), k=1)
            else:
                action = random.choices(np.arange(self.action_size), k=self.worker)
            return action



    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        if not self.munchausen:
            states, actions, rewards, next_states, dones = experiences
            # Get max predicted Q values (for next states) from target model
            Q_targets_next, _ = self.qnetwork_target(next_states, self.N) 
            Q_targets_next = Q_targets_next.detach().cpu()
            action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)
            Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)).transpose(1,2)
            # Compute Q targets for current states 
            Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
            # Get expected Q values from local model
            Q_expected, taus = self.qnetwork_local(states, self.N)
            Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
            
            loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
            loss = loss.mean()
        else:
            states, actions, rewards, next_states, dones = experiences
            Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
            Q_targets_next = Q_targets_next.detach() #(batch, num_tau, actions)
            q_t_n = Q_targets_next.mean(dim=1)

            # calculate log-pi 
            logsum = torch.logsumexp(\
                (q_t_n - q_t_n.max(1)[0].unsqueeze(-1))/self.entropy_tau, 1).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau*logsum).unsqueeze(1)
            
            pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1).unsqueeze(1)

            Q_target = (self.GAMMA**self.n_step * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.BATCH_SIZE, 1, self.N)

            q_k_target = self.qnetwork_target.get_qvalues(states).detach()
            v_k_target = q_k_target.max(1)[0].unsqueeze(-1) 
            tau_log_pik = q_k_target - v_k_target - self.entropy_tau*torch.logsumexp(\
                                                                    (q_k_target - v_k_target)/self.entropy_tau, 1).unsqueeze(-1)

            assert tau_log_pik.shape == (self.BATCH_SIZE, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.BATCH_SIZE, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            q_k, taus = self.qnetwork_local(states, self.N)
            Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))
            assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
            
            loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
            loss = loss.mean()


        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)

        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def learn_per(self, experiences):
            """Update value parameters using given batch of experience tuples.
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            self.optimizer.zero_grad()
            if not self.munchausen:
                states, actions, rewards, next_states, dones, idx, weights = experiences
                
                states = torch.FloatTensor(states).to(self.device)
                next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
                actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
                dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
                weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

                # Get max predicted Q values (for next states) from target model
                Q_targets_next, _ = self.qnetwork_target(next_states, self.N) 
                Q_targets_next = Q_targets_next.detach().cpu()
                action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)
                Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)).transpose(1,2)
                # Compute Q targets for current states 
                Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
                # Get expected Q values from local model
                Q_expected, taus = self.qnetwork_local(states, self.N)
                Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))

                # Quantile Huber loss
                td_error = Q_targets - Q_expected
                assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
                huber_l = calculate_huber_loss(td_error, 1.0)
                quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
                
                loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)* weights # , keepdim=True if per weights get multipl
                loss = loss.mean()
            else:
                states, actions, rewards, next_states, dones, idx, weights = experiences
                states = torch.FloatTensor(states).to(self.device)
                next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
                actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
                dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
                weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

                Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
                Q_targets_next = Q_targets_next.detach() #(batch, num_tau, actions)
                q_t_n = Q_targets_next.mean(dim=1)
                # calculate log-pi 
                logsum = torch.logsumexp(\
                    (Q_targets_next - Q_targets_next.max(2)[0].unsqueeze(-1))/self.entropy_tau, 2).unsqueeze(-1) #logsum trick
                assert logsum.shape == (self.BATCH_SIZE, self.N, 1), "log pi next has wrong shape"
                tau_log_pi_next = Q_targets_next - Q_targets_next.max(2)[0].unsqueeze(-1) - self.entropy_tau*logsum
                
                pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1).unsqueeze(1)

                Q_target = (self.GAMMA**self.n_step * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
                assert Q_target.shape == (self.BATCH_SIZE, 1, self.N)

                q_k_target = self.qnetwork_target.get_qvalues(states).detach()
                v_k_target = q_k_target.max(1)[0].unsqueeze(-1) # (8,8,1)
                tau_log_pik = q_k_target - v_k_target - self.entropy_tau*torch.logsumexp(\
                                                                        (q_k_target - v_k_target)/self.entropy_tau, 1).unsqueeze(-1)

                assert tau_log_pik.shape == (self.BATCH_SIZE, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
                munchausen_addon = tau_log_pik.gather(1, actions) #.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)
                
                # calc munchausen reward:
                munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
                assert munchausen_reward.shape == (self.BATCH_SIZE, 1, 1)
                # Compute Q targets for current states 
                Q_targets = munchausen_reward + Q_target
                # Get expected Q values from local model
                q_k, taus = self.qnetwork_local(states, self.N)
                Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))
                assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)

                # Quantile Huber loss
                td_error = Q_targets - Q_expected
                assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
                huber_l = calculate_huber_loss(td_error, 1.0)
                quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
                
                loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)* weights # , keepdim=True if per weights get multipl
                loss = loss.mean()


            # Minimize the loss
            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(),1)
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target)
            # update priorities
            td_error = td_error.sum(dim=1).mean(dim=1,keepdim=True) # not sure about this -> test 
            self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))
            return loss.detach().cpu().numpy()            

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    #assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss
    
            

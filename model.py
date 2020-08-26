import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias: 
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    
    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class IQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_step, seed, N, dueling=False, noisy=False, device="cuda:0"):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.state_dim = len(self.input_shape)
        self.action_size = action_size
        self.N = N  
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1,self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.dueling = dueling
        self.device = device
        if noisy:
            layer = NoisyLinear
        else:
            layer = nn.Linear

        # Network Architecture
        if self.state_dim == 3:
            self.head = nn.Sequential(
                nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            )#.apply() #weight init
            self.cos_embedding = nn.Linear(self.n_cos, self.calc_input_layer())
            self.ff_1 = layer(self.calc_input_layer(), layer_size)
            self.cos_layer_out = self.calc_input_layer()

        else:
            self.head = nn.Linear(self.input_shape[0], layer_size) 
            self.cos_embedding = nn.Linear(self.n_cos, layer_size)
            self.ff_1 = layer(layer_size, layer_size)
            self.cos_layer_out = layer_size
        if dueling:
            self.advantage = layer(layer_size, action_size)
            self.value = layer(layer_size, 1)
            #weight_init([self.head_1, self.ff_1])
        else:
            self.ff_2 = layer(layer_size, action_size)    
            #weight_init([self.head_1, self.ff_1])

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device) #(batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        
        x = torch.relu(self.head(input))
        if self.state_dim == 3: x = x.view(input.size(0), -1)
        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)
        
        x = torch.relu(self.ff_1(x))
        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, self.action_size), taus
    
    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.N)
        actions = quantiles.mean(dim=1)
        return actions  
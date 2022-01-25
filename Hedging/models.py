"""
Models -- Neural Networks
Policy and value function with fully-connected ANNs

"""
# numpy
import numpy as np
# pytorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# misc
import pdb # use with set_trace() for the debugger


# normalize features of the neural nets
def normalize_features(x, env):
    # normalize features with environment parameters
    x[...,0] = 2*x[...,0]/env.params["S0"] - 1.0 # price
    x[...,1] /= env.params["max_alpha"] # actual hedge position
    x[...,2] /= env.params["Ndt"] # time

    return x


# build a fully-connected neural net for the policy
class PolicyApprox(nn.Module):
    # constructor
    def __init__(self, input_size, env, n_layers, hidden_size, learn_rate=0.01):
        super(PolicyApprox, self).__init__()
        # input arguments
        self.input_size = input_size
        self.output_size = 1
        self.env = env
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learn_rate = learn_rate
        
        # build all layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)

        self.hidden_layers = []
        for i in range(self.n_layers-1):
            self.hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.layerN = nn.Linear(self.hidden_size, self.output_size)

        # initializers for weights and biases
        nn.init.normal_(self.layer1.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layer1.bias, 0)

        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(input_size)/2)
            nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.layerN.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layerN.bias, 0)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate) # SGD or Adam
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    # forward propagation
    def forward(self, x):
        # normalize features with environment parameters
        x = normalize_features(x, self.env)
        
        # mean of the Gaussian policy
        loc = F.silu(self.layer1(x))
        
        for layer in self.hidden_layers:
            loc = F.silu(layer(loc))

        # output layer attempts
        loc = T.clamp(self.layerN(loc),
                        min=-self.env.params["max_alpha"],
                        max=self.env.params["max_alpha"])

        # standard deviation of the Gaussian policy
        scale = 0.03

        return loc, scale

    # define parameters of the ANN
    def parameters(self):
        
        params = list(self.layer1.parameters())
        
        for layer in self.hidden_layers:
            params += list(layer.parameters())
            
        params += list(self.layerN.parameters())
        
        return params


# build a fully-connected neural net for the value function
class ValueApprox(nn.Module):
    # constructor
    def __init__(self, input_size, env, n_layers, hidden_size, learn_rate=0.01):
        super(ValueApprox, self).__init__()
        # input arguments
        self.input_size = input_size
        self.output_size = 1
        self.env = env
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learn_rate = learn_rate
        
        # build all layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)

        self.hidden_layers = []
        for i in range(self.n_layers-1):
            self.hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.layerN = nn.Linear(self.hidden_size, self.output_size)
        
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    # forward propagation
    def forward(self, x):
        # normalize features with environment parameters
        x = normalize_features(x, self.env)

        # value of the value function
        x = F.silu(self.layer1(x))
        
        for layer in self.hidden_layers:
            x = F.silu(layer(x))

        x = T.clamp(self.layerN(x), min=-30, max=30)

        return x

    # define parameters of the ANN
    def parameters(self):
        
        params = list(self.layer1.parameters())

        for layer in self.hidden_layers:
            params += list(layer.parameters())
            
        params += list(self.layerN.parameters())
        
        return params
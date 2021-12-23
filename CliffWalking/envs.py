"""
Environment


"""
# numpy
import numpy as np
# pytorch
import torch as T
# misc
import pdb # use with set_trace() for the debugger

class CliffEnv():
    # constructor
    def __init__(self, params):
        # parameters and spaces
        self.params = params
        self.spaces = {'t_space' : np.arange(params["T"]),
                      'pos_space' : np.linspace(-1, 12, 51),
                      'u_space' : np.linspace(-params["max_u"], params["max_u"], 41)}
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        pos = T.zeros(Nsims)
        
        return pos

    # initialization of the environment with multiple random states
    def random_reset(self, time, Nsims=1):
        temp = time*(time <= self.params["T"]/2) + self.params["T"]/2*(time > self.params["T"]/2)
        pos = temp*(self.params["max_u"]+2.3*self.params["sigma"])*T.rand(Nsims, device=self.device)
        
        return pos
    
    # simulation engine
    def step(self, pos_t, time_t, u_t):
        sizes = pos_t.shape
                
        pos_tp1 = pos_t + u_t
        time_tp1 = time_t + 1

        r = - self.params["C_time"]*T.ones(sizes) \
            - self.params["C_cliff"]*((pos_tp1 < self.params["cliff"]) & (time_tp1 != self.params["T"]-1)) \
            - self.params["C_move"](u_t)

        return pos_tp1, time_tp1, -r

    # terminal penalty on the inventory
    def get_final_reward(self, pos_T):
        r = -self.params["C_terminal"](pos_T)
        
        return -r

    ## NUMPY VERSIONS
    # simulation engine (numpy version)
    def step_np(self, pos_t, time_t, u_t):
        Nsims = pos_t.shape[0]

        pos_tp1 = pos_t + np.random.normal(u_t, self.params["sigma"])
        time_tp1 = time_t + 1

        r = - self.params["C_time"]*np.ones(Nsims) \
            - self.params["C_cliff"]*((pos_tp1 < self.params["cliff"]) & (time_tp1 != self.params["T"]-1)) \
            - self.params["C_move"](u_t)

        return pos_tp1, time_tp1, -r

    # terminal penalty on the inventory (numpy version)
    def get_final_reward_np(self, pos_T):
        r = -self.params["C_terminal"](pos_T)

        return -r
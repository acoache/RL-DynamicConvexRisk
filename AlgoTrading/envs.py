"""
Environment


"""
# numpy
import numpy as np
# pytorch
import torch as T
# misc
import pdb # use with set_trace() for the debugger

class TradingEnv():
    # constructor
    def __init__(self, params):
        # parameters and spaces
        self.params = params
        self.spaces = {'t_space' : np.arange(params["Ndt"]),
                      's_space' : np.linspace(params["theta"]-6*params["sigma"]/np.sqrt(2*params["kappa"]), 
                                  params["theta"]+6*params["sigma"]/np.sqrt(2*params["kappa"]), 51),
                      'q_space' : np.linspace(-params["max_q"], params["max_q"], 51),
                      'u_space' : np.linspace(-params["max_u"], params["max_u"], 21)}
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        s0 = T.normal(self.params["theta"],
                        self.params["sigma"]/np.sqrt(2*self.params["kappa"]),
                        size=(Nsims,),
                        device=self.device)
        q0 = T.zeros(Nsims)

        return s0, q0

    # initialization of the environment with multiple random states
    def random_reset(self, Nsims=1):
        s0 = T.normal(self.params["theta"],
                        4*self.params["sigma"]/np.sqrt(2*self.params["kappa"]),
                        size=(Nsims,),
                        device=self.device)
        s0 = T.maximum(s0, min(self.spaces["s_space"])*T.ones(1))
        q0 = -self.params["max_q"] + 2*self.params["max_q"]*T.rand(Nsims, device=self.device)

        return s0, q0
    
    # simulation engine
    def step(self, s_t, q_t, u_t):
        sizes = q_t.shape
        
        dt = self.params["T"]/self.params["Ndt"]
        sqrt_dt = np.sqrt(dt)
        eta = self.params["sigma"] * \
                np.sqrt((1 - np.exp(-2*self.params["kappa"]*dt)) / (2*self.params["kappa"]))
        
        # price modification - OU process
        s_tp1 = self.params["theta"] + \
                (s_t-self.params["theta"]) * np.exp(-self.params["kappa"]*dt) + \
                eta * T.randn(sizes, device=self.device)

        # inventory modification - add the trade to current inventory
        q_tp1 = q_t + u_t
        
        # reward - profit with transaction costs
        r = - s_t*u_t - self.params["phi"]*T.pow(u_t,2)
        
        return s_tp1, q_tp1, -r

    # terminal penalty on the inventory
    def get_final_cost(self, s_T, q_T):
        # reward - profit with terminal penalty
        r = q_T*s_T - self.params["psi"]*T.pow(q_T,2)
        
        return -r

    ## NUMPY VERSIONS
    # simulation engine (numpy version)
    def step_np(self, s_t, q_t, u_t):
        Nsims = q_t.shape[0]
        
        dt = self.params["T"]/self.params["Ndt"]
        sqrt_dt = np.sqrt(dt)
        eta = self.params["sigma"] * \
                np.sqrt((1 - np.exp(-2*self.params["kappa"]*dt)) / (2*self.params["kappa"]))
        
        # price modification - OU process
        s_tp1 = self.params["theta"] + \
                (s_t-self.params["theta"]) * np.exp(-self.params["kappa"]*dt) + \
                eta * np.random.randn(Nsims)

        # inventory modification - add the trade to current inventory
        q_tp1 = q_t + u_t

        # reward - profit with transaction costs
        r = - s_t*u_t - self.params["phi"]*(u_t**2)
        
        return s_tp1, q_tp1, -r

    # terminal penalty on the inventory (numpy version)
    def get_final_cost_np(self, s_T, q_T):
        # reward - profit with terminal penalty
        r = q_T*s_T - self.params["psi"]*(q_T**2)
        
        return -r
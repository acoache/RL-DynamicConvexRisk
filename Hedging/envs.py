"""
Environment


"""
# numpy
import numpy as np
# pytorch
import torch as T
# misc
import pdb # use with set_trace() for the debugger


class HedgingEnv():
    # constructor
    def __init__(self, params):
        # parameters and spaces
        self.params = params
        self.spaces = {'t_space' : np.arange(params["Ndt"]),
                      'S_space' : np.linspace(params["S0"]*np.exp(-1/2*params["theta"]*params["T"]+np.sqrt(params["theta"]*params["T"])*-3),
                                              params["S0"]*np.exp(-1/2*params["theta"]*params["T"]+np.sqrt(params["theta"]*params["T"])*3), 31),
                      'v_space' : np.linspace(0.0, 2*params["eta"]*params["v0"], 31),
                      'alpha_space' : np.linspace(-params["max_alpha"], params["max_alpha"], 31),
                      'B_space' : np.linspace(params["B0"]-params["S0"]*params["max_alpha"],
                                            params["B0"]+params["S0"]*params["max_alpha"], 31)}
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        S0 = self.params["S0"]*T.ones(Nsims) # price
        v0 = self.params["v0"]*T.ones(Nsims) # volatility
        alpha_m1 = T.zeros(Nsims) # amount of the stock
        B0 = self.params["B0"]*T.ones(Nsims) # bank account cash-flow

        return S0, v0, alpha_m1, B0


    # initialization of the environment with multiple random states
    def random_reset(self, time, Nsims=1):
        if time == self.spaces["t_space"][0]:
            S0, v0, alpha_m1, B0 = self.reset(Nsims)
        else:
            S0 = self.params["S0"] * \
                    T.exp(-1/2*self.params["theta"]*self.params["T"] + \
                        np.sqrt(self.params["theta"]*self.params["T"]) * T.randn(size=(Nsims,), device=self.device))
            v0 = T.maximum(self.params["theta"] + \
                            self.params["eta"] * np.sqrt(self.params["theta"]) /  \
                            np.sqrt(2*self.params["kappa"]) * T.randn(size=(Nsims,), device=self.device),
                            0.001*T.ones(1))
            alpha_m1 = -self.params["max_alpha"] + 2*self.params["max_alpha"]*T.rand(size=(Nsims,), device=self.device)
            B0 = -(self.params["S0"]*alpha_m1) + 0.75*T.randn(size=(Nsims,), device=self.device)

        return S0, v0, alpha_m1, B0
    

    # simulation engine
    def step(self, S_t, v_t, alpha_tm1, B_t, alpha_t):
        # S_t : price of the stock
        # v_t : volatility of the stock
        # alpha_tm1 : amount of the stock held by the agent
        # B_t : bank account cash-flow
        # alpha_t : new amount of the stock held by the agent (action)

        Nsims = list(S_t.shape)
        Nsims.append(2)
        
        # time intervals
        dt = self.params["T"]/self.params["Ndt"]
        sqrt_dt = np.sqrt(dt)
        vp = T.maximum(v_t, T.zeros(1))

        # correlation matrix
        Omega = T.Tensor([[1, self.params["rho"]], [0, np.sqrt(1-self.params["rho"]**2)]])
        
        # generate correlated brownian motions
        W = T.einsum('...k,kl->...l', T.randn((Nsims), device=self.device), Omega)
        
        # simulate the price
        S_tp1 = S_t * T.exp((self.params["mu"]-0.5*vp)*dt + sqrt_dt*T.sqrt(vp)*W[...,0])
        
        # simulate the volatility
        v_tp1 = v_t + self.params["kappa"] * (self.params["theta"] - vp) * dt \
                    + self.params["eta"] * sqrt_dt * T.sqrt(vp) * W[...,1] \
                    + (v_t >= 0)*(0.25 * self.params["eta"]**2 * dt * (W[...,1]**2 - 1))

        # compute the bank account cash-flow
        B_tplus = B_t \
                - (alpha_t-alpha_tm1)*S_t \
                - T.abs(alpha_t-alpha_tm1)*self.params["epsilon"]

        # interest rate on the bank account
        B_tp1 = B_tplus * np.exp(self.params["r"]*dt)

        # compute the portfolio cash-flow
        W_t = B_t + alpha_tm1*S_t
        W_tp1 = B_tp1 + alpha_t*S_tp1

        # calculate the reward
        r = W_tp1 - W_t
        
        return S_tp1, v_tp1, alpha_t, B_tp1, -r


    # terminal penalty based on inventory and call option
    def get_final_cost(self, S_T, v_T, alpha_Tm1, B_T):
        r = - T.abs(alpha_Tm1)*self.params["epsilon"] - self.option_price(S_T)
        
        return -r


    # payoff of the option
    def option_price(self, S):
        return T.maximum(S - self.params["K"], T.zeros(1))
"""
Main -- Algorithmic Trading Problem
Value function & policy represented by a single ANN
Value function is learned from the current policy
"""
# numpy
import numpy as np
import numpy.matlib
# plotting
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
# pytorch
import torch as T
import torch.optim as optim
# personal files
from utils
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import TradingEnv
from actor_critic import ActorCriticPG
# misc
import time
import os
import pdb # use with set_trace() for the debugger

"""
Parameters
"""

# running on a personal computer or a Compute Canada server
computer = 'personal' # 'cluster' | 'personal'
repo_name = 'AlgoTrading_ex1'

# risk measures used
rm_list = ['mean', 'CVaR', 'CVaR-penalized'] # 'mean','CVaR','semi-dev','CVaR-penalized','mean-CVaR'
alpha_cvar = [-99, 0.2, 0.2] # threshold for the conditional value-at-risk
kappa_semidev = [-99, -99, 0.2] # coefficient for the mean semideviation
r_semidev = [-99, -99, -99] # exponent of the mean-semideviation

# parameters for the model
params = {'kappa' : 2, # kappa of the OU process
          'sigma' : 0.2, # standard deviation of the OU process
          'theta' : 1, # mean-reversion level of the OU process
          'phi' : 0.005, # transaction costs
          'psi' : 0.5, # terminal penalty on the inventory
          'T' : 1, # trading horizon
          'Ndt' : 5+1, # number of periods
          'max_q' : 5, # maximum value for the inventory
          'max_u' : 2} # maximum value for the trades

# training phase parameters
Ntrajectories = 500 # number of generated trajectories
Mtransitions = 2000 # number of additional transitions for each state
Nepochs = 300 # number of epochs of the whole algorithm
gamma = 1.00 # discount factor
print_progress = 200 # number of epochs before printing the time
plot_policy = 50 # number of epochs before plotting the policy/value function

Nepochs_V_init = 500 # number of epochs for the estimation of V during the first epoch
Nepochs_V = 50 # number of epochs for the estimation of V
lr_V = 1e-3 # learning rate of the neural net associated with V
batch_V = 300 # number of trajectories for each mini-batch in estimating V
hidden_V = 16 # number of hidden nodes in the neural net associated with V

Nepochs_pi = 10 # number of epoch for the update of pi
lr_pi = 1e-3 # learning rate of the neural net associated with pi
batch_pi = 300 # number of trajectories for each mini-batch when updating pi
hidden_pi = 16 # number of hidden nodes in the neural net associated with pi

seed = 4321 # set seed for replication purposes

Nsims_optimal = 1000 # number of simulations when using the brute force method

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')
print('*  T: ', params["T"],
        ' Ndt: ', params["Ndt"],
        ' b: ', params["b"],
        ' kappa: ', params["kappa"],
        ' sigma: ', params["sigma"],
        ' theta: ', params["theta"],
        ' phi: ', params["phi"],
        ' psi: ', params["psi"],
        ' max_q: ', params["max_q"],
        ' max_u: ', params["max_u"])
print('*  Ntrajectories: ', Ntrajectories,
        ' Mtransitions: ', Mtransitions, 
        ' Nepochs: ', Nepochs,
        ' Nsims_optimal: ', Nsims_optimal)
print('*  Nepochs_V_init: ', Nepochs_V_init,
        ' Nepochs_V: ', Nepochs_V,
        ' lr_V: ', lr_V, 
        ' batch_V: ', batch_V,
        ' hidden_V: ', hidden_V)
print('*  Nepochs_pi: ', Nepochs_pi,
        ' lr_pi: ', lr_pi, 
        ' batch_pi: ', batch_pi,
        ' hidden_pi: ', hidden_pi)
print('*  alpha_cvar: ', alpha_cvar,
        ' kappa_semidev: ', kappa_semidev,
        ' r_semidev: ', r_semidev)

# create a new directory
if(computer == 'personal'):
    repo = repo_name
if(computer == 'cluster'):
    # get enviroment directories
    data_dir = os.getenv("HOME")
    output_dir = os.getenv("SCRATCH")
    repo = output_dir + '/' + repo_name

utils.directory(repo)

# loop for all risk measures
for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the environment and risk measure objects
    env = TradingEnv(params)
    risk_measure = RiskMeasure(Type=method,
                                alpha=alpha_cvar[idx_method],
                                kappa=kappa_semidev[idx_method],
                                r=r_semidev[idx_method])

    # create repositories
    if(method == 'CVaR'):
        method = method + str( round(alpha_cvar[idx_method],3) )
    if(method == 'semi-dev'):
        method = method + str( round(kappa_semidev[idx_method],3) )
    if(method == 'mean-CVaR'):
        method = 'mean' + str(round(kappa_semidev[idx_method],3)) \
                    + '-CVaR' + str(round(alpha_cvar[idx_method],3))
    if(method == 'CVaR-penalized'):
        method = 'CVaR' + str(round(alpha_cvar[idx_method],3)) \
                    + '-pen' + str(round(kappa_semidev[idx_method],3))

    utils.directory(repo + '/' + method)
    for time_idx in env.spaces["t_space"][:-1][::-1]:
        utils.directory(repo + '/' + method + '/time' + str(time_idx+1))

    # create policy & value function objects
    # single neural network; (price x inventory x time)
    policy = PolicyApprox(3, env, hidden_size=hidden_pi, learn_rate=lr_pi)
    value_function = ValueApprox(3, env, hidden_size=hidden_V, learn_rate=lr_V)
    
    # initialize the actor-critic algorithm
    actor_critic = ActorCriticPG(repo=repo,
                                    method = method,
                                    env=env,
                                    policy=policy,
                                    V=value_function,
                                    risk_measure=risk_measure,
                                    gamma=gamma)

    # obtain the optimal value function and policy
    actor_critic.optimal_policy(Nsims=Nsims_optimal, plot=True)

    ## TRAINING PHASE
    # first estimate of the value function
    actor_critic.estimate_V(Ntrajectories=Ntrajectories,
                                Mtransitions=Mtransitions,
                                batch_size=batch_V,
                                Nepochs=Nepochs_V_init,
                                rng_seed=None)

    # plot current policy
    actor_critic.plot_current_Vs()
    actor_critic.plot_current_policies()

    for epoch in range(Nepochs):
        # obtain a seed
        rng_seed = None
        # rng_seed = T.seed()

        # estimate the value function of the current policy
        actor_critic.estimate_V(Ntrajectories=Ntrajectories,
                                    Mtransitions=Mtransitions,
                                    batch_size=batch_V,
                                    Nepochs=Nepochs_V,
                                    rng_seed=rng_seed)
        
        # update the policy by policy gradient
        actor_critic.update_policy(Ntrajectories=Ntrajectories,
                                    Mtransitions=Mtransitions,
                                    batch_size=batch_pi,
                                    Nepochs=Nepochs_pi,
                                    rng_seed=rng_seed)

        # print progress
        if epoch % print_progress == 0 or epoch == Nepochs - 1:
            print('*** Epoch = ', str(epoch) ,
                    ' completed, Duration = ', "{:.3f}".format(time.time() - start_time), ' secs ***')
            start_time = time.time()

        # plot current policy
        if epoch % plot_policy == 0 or epoch == Nepochs - 1:
            actor_critic.plot_current_Vs()
            actor_critic.plot_current_policies()

    # save the neural network
    T.save(actor_critic.policy.state_dict(),
            repo + '/' + method + '/policy_model.pt')
    T.save(actor_critic.V.state_dict(),
            repo + '/' + method + '/V_model.pt')
    # to load the model, M = ModelClass(*args, **kwargs); M.load_state_dict(T.load(PATH))

    # print progress
    print('*** Training phase completed! ***')
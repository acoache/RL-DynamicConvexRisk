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
import utils
import hyperparams
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import TradingEnv
from actor_critic import ActorCriticPG
# misc
import time
import os
import pdb # use with set_trace() for the debugger
from datetime import datetime

"""
Parameters
"""

# running on a personal computer or a Compute Canada server
computer = 'personal' # 'cluster' | 'personal'
preload = False # load pre-trained model prior to the training phase

# risk measures used
rm_list = ['mean', 'CVaR', 'CVaR-penalized'] # 'mean' | 'CVaR' | 'semi-dev' | 'CVaR-penalized' | 'mean-CVaR'
alpha_cvar = [-99, 0.2, 0.2] # threshold for the conditional value-at-risk
kappa_semidev = [-99, -99, 0.2] # coefficient for the mean semideviation
r_semidev = [-99, -99, -99] # exponent of the mean-semideviation

# parameters for the model and algorithm
repo_name, envParams, algoParams = hyperparams.initParams()

print_progress = 200 # number of epochs before printing the time/loss
plot_progress = 50 # number of epochs before plotting the policy/value function
save_progress = 100 # number of epochs before saving the policy/value function ANNs
                    
"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')
hyperparams.printParams(envParams, algoParams)
print('*  alpha_cvar: ', alpha_cvar,
        ' kappa_semidev: ', kappa_semidev,
        ' r_semidev: ', r_semidev)

# create a new directory
if(computer == 'personal'): # personal computer
    repo = repo_name
    data_repo = repo_name
if(computer == 'cluster'): # Compute Canada server
    data_dir = os.getenv("HOME")
    output_dir = os.getenv("SCRATCH")
    repo = output_dir + '/' + repo_name
    data_repo = data_dir + '/ComputeCanadaRepo/' + repo_name

utils.directory(repo)

# loop for all risk measures
for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the environment and risk measure objects
    env = TradingEnv(envParams)
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
    policy = PolicyApprox(3, env,
                            n_layers=algoParams["layers_pi"],
                            hidden_size=algoParams["hidden_pi"],
                            learn_rate=algoParams["lr_pi"])
    value_function = ValueApprox(3, env,
                            n_layers=algoParams["layers_V"],
                            hidden_size=algoParams["hidden_V"],
                            learn_rate=algoParams["lr_V"])
    
    # initialize the actor-critic algorithm
    actor_critic = ActorCriticPG(repo=repo,
                                    method = method,
                                    env=env,
                                    policy=policy,
                                    V=value_function,
                                    risk_measure=risk_measure,
                                    gamma=algoParams["gamma"])

    # obtain the optimal value function and policy
    actor_critic.optimal_policy(Nsims=algoParams["Nsims_optimal"], plot=True)

    if preload:
        # load the weights of the pre-trained model
        actor_critic.policy.load_state_dict(T.load(data_repo + '/' + method + '/policy_model.pt'))
        actor_critic.V.load_state_dict(T.load(data_repo + '/' + method + '/V_model.pt'))

    ## TRAINING PHASE
    # first estimate of the value function
    actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                                Mtransitions=algoParams["Mtransitions"],
                                batch_size=algoParams["batch_V"],
                                Nepochs=algoParams["Nepochs_V_init"],
                                rng_seed=algoParams["seed"])

    # plot current policy
    actor_critic.plot_current_Vs()
    actor_critic.plot_current_policies()

    for epoch in range(algoParams["Nepochs"]):
        # estimate the value function of the current policy
        actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                                    Mtransitions=algoParams["Mtransitions"],
                                    batch_size=algoParams["batch_V"],
                                    Nepochs=algoParams["Nepochs_V"],
                                    rng_seed=algoParams["seed"])
        
        # update the policy by policy gradient
        actor_critic.update_policy(Ntrajectories=algoParams["Ntrajectories"],
                                    Mtransitions=algoParams["Mtransitions"],
                                    batch_size=algoParams["batch_pi"],
                                    Nepochs=algoParams["Nepochs_pi"],
                                    rng_seed=algoParams["seed"])

        # print progress
        if epoch % print_progress == 0 or epoch == algoParams["Nepochs"] - 1:
            print('*** Epoch = ', str(epoch) ,
                    ' completed, Duration = ', "{:.3f}".format(time.time() - start_time), ' secs ***')
            start_time = time.time()

        # plot current policy
        if epoch % plot_progress == 0 or epoch == algoParams["Nepochs"] - 1:
            actor_critic.plot_current_Vs()
            actor_critic.plot_current_policies()

        # save progress
        if epoch % save_progress == 0:
            now = datetime.now()
            # save the neural network
            T.save(actor_critic.policy.state_dict(),
                    repo + '/' + method + '/policy_model' + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.pt')
            T.save(actor_critic.V.state_dict(),
                    repo + '/' + method + '/V_model' + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.pt')

    # save the neural network
    T.save(actor_critic.policy.state_dict(),
            repo + '/' + method + '/policy_model.pt')
    T.save(actor_critic.V.state_dict(),
            repo + '/' + method + '/V_model.pt')
    # to load the model, M = ModelClass(*args, **kwargs); M.load_state_dict(T.load(PATH))

    # print progress
    print('*** Training phase completed! ***')
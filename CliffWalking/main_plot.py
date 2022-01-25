"""
Plots -- Cliff Walking Problem
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
from envs import CliffEnv
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

# risk measures used
# 'mean' | 'CVaR' | 'semi-dev' | 'CVaR-penalized' | 'mean-CVaR'
rm_list = ['mean', 'CVaR0.2', 'CVaR0.2-pen0.2']

# parameters for the model and algorithm
repo_name, envParams, algoParams = hyperparams.initParams()

seed = 4321 # set seed for replication purposes

# testing phase parameters
Nsimulations = 30000 # number of simulations following the optimal strategy

# font sizes for figures
plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=20)

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')
hyperparams.printParams(envParams, algoParams)

# create a new directory
if(computer == 'personal'): # personal computer
    repo = repo_name
if(computer == 'cluster'): # Compute Canada server
    data_dir = os.getenv("HOME")
    output_dir = os.getenv("SCRATCH")
    repo = output_dir + '/' + repo_name

utils.directory(repo)

costs = np.zeros((Nsimulations, envParams["T"], len(rm_list))) # matrix to store all testing trajectories
positions = np.zeros((Nsimulations, envParams["T"], len(rm_list))) # matrix to store all testing trajectories


for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the (temporary) environment and risk measure objects
    env = CliffEnv(envParams)
    risk_measure = RiskMeasure(Type='mean')

    # create policy & value function objects
    # single neural network; (position x time)
    policy = PolicyApprox(2, env,
                            n_layers=algoParams["layers_pi"],
                            hidden_size=algoParams["hidden_pi"],
                            learn_rate=algoParams["lr_pi"])
    value_function = ValueApprox(2, env,
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

    # load the trained model
    actor_critic.policy.load_state_dict(T.load(repo + '/' + method + '/policy_model.pt'))
    actor_critic.V.load_state_dict(T.load(repo + '/' + method + '/V_model.pt'))

    # print progress
    print('*** Training phase completed! ***')

    ## TESTING PHASE
    # set seed for reproducibility purposes
    T.manual_seed(seed)
    np.random.seed(seed)

    # initialize the starting state
    pos = env.reset(Nsimulations)
    positions[:,0,idx_method] = pos.detach().numpy()
    
    for timestep in env.spaces["t_space"][:-1]:
        # simulate transitions according to the policy
        u, _ = actor_critic.select_actions(pos, timestep*T.ones(Nsimulations), 'random')
        pos, t_tp1, cost = env.step(pos, timestep*np.ones(Nsimulations), u)
        
        # store states and costs
        positions[:,timestep+1,idx_method] = pos.detach().numpy()
        costs[:,timestep,idx_method] = cost.detach().numpy()

    # get terminal reward
    costs[:,-1,idx_method] = env.get_final_cost(pos).detach().numpy()

    ### PLOT - policy with the paths
    # initialize 2D histogram
    hist2dim_pi = np.zeros([len(env.spaces["pos_space"]), len(env.spaces["t_space"])-1])
    
    for pos_idx, pos_val in enumerate(env.spaces["pos_space"]):
        for time_idx, time_val in enumerate(env.spaces["t_space"][:-1]):
            # mean of the Gaussian policy
            hist2dim_pi[len(env.spaces["pos_space"])-pos_idx-1, time_idx], _ = \
                    actor_critic.select_actions(T.Tensor([pos_val]),
                                                T.tensor([time_val]),
                                                'best')

    # plot the 2D histogram
    plt.imshow(hist2dim_pi,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(env.spaces["t_space"]), 
                        np.max(env.spaces["t_space"]),
                        np.min(env.spaces["pos_space"]),
                        np.max(env.spaces["pos_space"])],
                aspect='auto',
                vmin=-env.params["max_u"],
                vmax=env.params["max_u"])

    # plot the cliff
    plt.axhline(y=env.params["cliff"],
                color='black',
                linestyle='--',
                linewidth=1.0)

    # plot quantiles of the different paths
    plt.plot(np.arange(env.params["T"]),
            np.quantile(positions[:,:,idx_method],0.1, axis=0),
            linestyle='-',
            color=utils.mgreen,
            linewidth=1.5)
    plt.plot(np.arange(env.params["T"]),
            np.quantile(positions[:,:,idx_method],0.5, axis=0),
            linestyle='-',
            color=utils.mgreen,
            linewidth=1.5)
    plt.plot(np.arange(env.params["T"]),
            np.quantile(positions[:,:,idx_method],0.9, axis=0),
            linestyle='-',
            color=utils.mgreen,
            linewidth=1.5)

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Learned Policy")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(repo + '/learnedpolicy_' + method + '_paths.pdf', transparent=True)
    plt.clf()

"""
# Plots & figures
"""

# plot rewards instead of costs
rewards_total = -1 * np.sum(costs, axis=1)

# set a grid for the histogram
grid = np.linspace(np.min(rewards_total), np.max(rewards_total), 100)

### PLOT - Distribution of the terminal reward
for idx_method, method in enumerate(rm_list):
    # plot the histogram for each method
    reward = rewards_total[:,idx_method]
    plt.hist(x=rewards_total[:,idx_method],
            alpha=0.4,
            bins=grid,
            color=utils.colors[idx_method],
            density=True)

plt.legend(rm_list)
plt.xlabel("Terminal reward")
plt.ylabel("Density")
plt.title("Distribution of the terminal reward")

for idx_method, method in enumerate(rm_list):
    # plot gaussian KDEs
    kde = gaussian_kde(rewards_total[:,idx_method], bw_method='silverman')
    plt.plot(grid,
            kde(grid),
            color=utils.colors[idx_method],
            linewidth=1.5)
    # plot quantiles of the distributions
    plt.axvline(x=np.quantile(rewards_total[:,idx_method],0.1),
                linestyle='dashed',
                color=utils.colors[idx_method],
                linewidth=1.0)
    # plt.axvline(x=np.mean(rewards_total[:,idx_method]),
    #             linestyle='dotted',
    #             color=utils.colors[idx_method],
    #             linewidth=1.0)
    plt.axvline(x=np.quantile(rewards_total[:,idx_method],0.9),
                linestyle='dashed',
                color=utils.colors[idx_method],
                linewidth=1.0)

plt.tight_layout()
plt.savefig(repo + '/comparison_terminal_cost.pdf', transparent=True)
plt.clf()

### PLOT - Paths of the rover over several simulations
for idx_method, method in enumerate(rm_list):
    # plot median of the different paths
    plt.plot(np.arange(env.params["T"]),
            np.quantile(positions[:,:,idx_method],0.5, axis=0),
            linestyle='-',
            color=utils.colors[idx_method],
            linewidth=1.0)

plt.legend(rm_list)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("")

# plot the cliff
plt.axhline(y=env.params["cliff"],
    color='black',
    linestyle='--',
    linewidth=1.0)

for idx_method, method in enumerate(rm_list):
    # plot quantiles of the different paths
    plt.plot(np.arange(env.params["T"]),
            np.quantile(positions[:,:,idx_method],0.1, axis=0),
            linestyle='-',
            color=utils.colors[idx_method],
            linewidth=1.0)
    plt.plot(np.arange(env.params["T"]),
            np.quantile(positions[:,:,idx_method],0.9, axis=0),
            linestyle='-',
            color=utils.colors[idx_method],
            linewidth=1.0)
    # plt.fill_between(np.arange(env.params["T"]),
    #                 np.quantile(positions[:,:,idx_method],0.1, axis=0),
    #                 np.quantile(positions[:,:,idx_method],0.9, axis=0),
    #                 color=utils.colors[idx_method],  
    #                 alpha=0.4)

plt.tight_layout()
plt.savefig(repo + '/comparison_paths.pdf', transparent=True)
plt.clf()

# print progress
print('*** Testing phase completed! ***')
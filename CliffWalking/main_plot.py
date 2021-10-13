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
repo_name = 'CliffWalking_ex1'

# risk measures used
# 'mean' | 'CVaR' | 'semi-dev' | 'CVaR-penalized' | 'mean-CVaR'
# rm_list = ['mean', 'CVaR0.2', 'CVaR0.2-pen0.1', 'CVaR0.2-pen0.5']
# rm_list = ['mean', 'CVaR0.5', 'CVaR0.5-pen0.1', 'CVaR0.5-pen0.5']
rm_list = ['mean', 'CVaR0.5', 'CVaR0.2', 'CVaR0.1']

def cost_move_t(pos):
    return abs(pos)**2

def cost_move_T(pos):
    return abs(pos)**2

# parameters for the model
params = {'T' : 9, # number of periods
          'cliff' : 1.0, # position of the cliff
          'C_cliff' : 100, # cost when falling into the cliff
          'C_time' : 1.0, # cost from a period to another
          'C_move' : cost_move_t, # cost from a movement
          'C_terminal' : cost_move_T, # terminal penalty on the position
          'max_u' : 4.0, # maximum of the mean of the Gaussian policy
          'sigma': 1.50} # standard deviation of the Gaussian policy

# training phase parameters
gamma = 1.00 # discount factor

Nepochs_V = 1 # number of epochs for the estimation of V
lr_V = 1e-3 # learning rate of the neural net associated with V
batch_V = 100 # number of trajectories for each mini-batch in estimating V
hidden_V = 16 # number of hidden nodes in the neural net associated with V

Nepochs_pi = 1 # number of epoch for the update of pi
lr_pi = 1e-3 # learning rate of the neural net associated with pi
batch_pi = 100 # number of trajectories for each mini-batch when updating pi
hidden_pi = 16 # number of hidden nodes in the neural net associated with pi

seed = 4321 # set seed for replication purposes

# testing phase parameters
Nsimulations = 30000 # number of simulations following the optimal strategy

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')
print('*  T: ', params["T"],
        ' cliff: ', params["cliff"],
        ' C_cliff: ', params["C_cliff"],
        ' max_u: ', params["max_u"],
        ' sigma: ', params["sigma"])

# create a new directory
if(computer == 'personal'): # personal computer
    repo = repo_name
if(computer == 'cluster'): # Compute Canada server
    data_dir = os.getenv("HOME")
    output_dir = os.getenv("SCRATCH")
    repo = output_dir + '/' + repo_name

utils.directory(repo)

rewards = np.zeros((Nsimulations, params["T"], len(rm_list))) # matrix to store all testing trajectories
positions = np.zeros((Nsimulations, params["T"], len(rm_list))) # matrix to store all testing trajectories


for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the (temporary) environment and risk measure objects
    env = TradingEnv(params)
    risk_measure = RiskMeasure(Type='mean')

    # create policy & value function objects
    # single neural network; (position x time)
    policy = PolicyApprox(2, env, hidden_size=hidden_pi, learn_rate=lr_pi)
    value_function = ValueApprox(2, env, hidden_size=hidden_V, learn_rate=lr_V)

    # initialize the actor-critic algorithm
    actor_critic = ActorCriticPG(repo=repo,
                                    method = method,
                                    env=env,
                                    policy=policy,
                                    V=value_function,
                                    risk_measure=risk_measure,
                                    gamma=gamma)
    actor_critic.policy.load_state_dict(T.load(repo + '/' + method + '/policy_model.pt'))
    actor_critic.V.load_state_dict(T.load(repo + '/' + method + '/V_model.pt'))

    # print progress
    print('*** Training phase completed! ***')

    ## TESTING PHASE
    # set seed for reproducibility purposes
    np.random.seed(seed)

    # initialize the starting state
    pos = env.reset(Nsimulations)
    positions[:,0,idx_method] = pos.detach().numpy()
    
    for t_t in env.spaces["t_space"][:-1]:
        # simulate transitions according to the policy
        u, _ = actor_critic.select_actions(pos, t_t*T.ones(Nsimulations), 'random')
        pos, t_tp1, r = env.step(pos, t_t*np.ones(Nsimulations), u)
        
        # store states and costs
        positions[:,t_t+1,idx_method] = pos.detach().numpy()
        rewards[:,t_t,idx_method] = r.detach().numpy()

    # get terminal reward
    rewards[:,-1,idx_method] = env.get_final_reward(pos).detach().numpy()

"""
# Plots & figures
"""

# plot rewards instead of costs
Total_r = -1 * np.sum(rewards, axis=1)

# set a grid for the histogram
grid = np.linspace(np.min(Total_r), np.max(Total_r), 100)

### PLOT 1 - Distribution of the terminal reward
for idx_method, method in enumerate(rm_list):
    # plot the histogram for each method
    reward = Total_r[:,idx_method]
    plt.hist(x=Total_r[:,idx_method],
            alpha=0.4,
            bins=grid,
            color=utils.colors[idx_method],
            density=True)

plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=20)
plt.legend(rm_list)
plt.xlabel("Terminal reward")
plt.ylabel("Density")
plt.title("Distribution of the terminal reward")

for idx_method, method in enumerate(rm_list):
    # plot gaussian KDEs
    kde = gaussian_kde(Total_r[:,idx_method], bw_method='silverman')
    plt.plot(grid,
            kde(grid),
            color=utils.colors[idx_method],
            linewidth=1.5)
    # plot quantiles of the distributions
    plt.axvline(x=np.quantile(Total_r[:,idx_method],0.1),
                linestyle='dashed',
                color=utils.colors[idx_method],
                linewidth=1.0)
    # plt.axvline(x=np.mean(Total_r[:,idx_method]),
    #             linestyle='dotted',
    #             color=utils.colors[idx_method],
    #             linewidth=1.0)
    plt.axvline(x=np.quantile(Total_r[:,idx_method],0.9),
                linestyle='dashed',
                color=utils.colors[idx_method],
                linewidth=1.0)

plt.tight_layout()
plt.savefig(repo + '/comparison_terminal_cost.pdf', transparent=True)
plt.clf()

### PLOT 2 - Paths of the rover over several simulations
for idx_method, method in enumerate(rm_list):
    # plot median of the different paths
    plt.plot(np.arange(env.params["T"]),
            np.quantile(positions[:,:,idx_method],0.5, axis=0),
            linestyle='-',
            color=utils.colors[idx_method],
            linewidth=1.0)

plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=20)
plt.legend(rm_list)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("")

# plot the cliff
plt.axhline(y=params["cliff"],
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

### PLOT 3 - policy with the paths
for idx_method, method in enumerate(rm_list):
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
    plt.axhline(y=params["cliff"],
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

    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', labelsize=20)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Learned Policy")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(repo + '/learnedpolicy_' + method + '_paths.pdf', transparent=True)
    plt.clf()

# print progress
print('*** Testing phase completed! ***')
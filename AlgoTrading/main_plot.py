"""
Plots -- Algorithmic Trading Problem
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

# risk measures used (full name for repository)
# 'mean','CVaR','semi-dev','CVaR-penalized','mean-CVaR'
# rm_list = ['mean', 'CVaR0.2', 'CVaR0.2-pen0.1', 'CVaR0.2-pen0.5']
# rm_list = ['mean', 'CVaR0.5', 'CVaR0.5-pen0.1', 'CVaR0.5-pen0.5']
rm_list = ['mean', 'CVaR0.5', 'CVaR0.2', 'CVaR0.1']

# parameters for the model
params = {'b' : 0.1,
          'kappa' : 2,
          'sigma' : 0.2,
          'theta' : 1,
          'phi' : 0.005,
          'psi' : 0.5,
          'T' : 1,
          'Ndt' : 5+1,
          'max_q' : 5,
          'max_u' : 2}

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
        ' Ndt: ', params["Ndt"],
        ' b: ', params["b"],
        ' kappa: ', params["kappa"],
        ' sigma: ', params["sigma"],
        ' theta: ', params["theta"],
        ' phi: ', params["phi"],
        ' psi: ', params["psi"],
        ' max_q: ', params["max_q"],
        ' max_u: ', params["max_u"])

# create a new directory
if(computer == 'personal'):
    repo = repo_name
if(computer == 'cluster'):
    # get enviroment directories
    data_dir = os.getenv("HOME")
    output_dir = os.getenv("SCRATCH")
    repo = output_dir + '/' + repo_name

utils.directory(repo)

simulations = np.zeros((Nsimulations, params["Ndt"], len(rm_list))) # matrix to store all testing trajectories


for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the environment and risk measure objects
    env = TradingEnv(params)
    risk_measure = RiskMeasure(Type='mean')

    # create policy & value function objects
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

    # load the trained model
    actor_critic.policy.load_state_dict(T.load(repo + '/' + method + '/policy_model.pt'))

    # print progress
    print('*** Training phase completed! ***')

    ## TESTING PHASE
    # distribution of terminal reward under the optimal policy
    np.random.seed(seed)
    s, q = env.reset(Nsimulations)
    for timestep in env.spaces["t_space"][:-1]:
        u, _ = actor_critic.select_actions(s, q, timestep*T.ones(Nsimulations), 'best')
        s, q, r = env.step(s, q, u)
        simulations[:,timestep,idx_method] = r.detach().numpy()

    simulations[:,-1,idx_method] = env.get_final_reward(s, q).detach().numpy()

"""
# Plots & figures
"""
# plot rewards instead of costs
Total_r = -1 * np.sum(simulations, axis=1)

# set a grid for the histogram
grid = np.linspace(np.min(Total_r), np.max(Total_r), 100)

### PLOT 1 - Distribution of the terminal reward
for idx_method, method in enumerate(rm_list):
    # compare both terminal reward distribution
    reward = Total_r[:,idx_method]
    plt.hist(x=Total_r[:,idx_method],
            alpha=0.4,
            bins=grid,
            color=utils.colors[idx_method],
            density=True)

plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=20)
plt.legend(rm_list)
plt.xlabel("Terminal wealth")
plt.ylabel("Density")
plt.title("Distribution of the terminal wealth")

for idx_method, method in enumerate(rm_list):
    # gaussian KDEs
    kde = gaussian_kde(Total_r[:,idx_method], bw_method='silverman')
    plt.plot(grid,
            kde(grid),
            color=utils.colors[idx_method],
            linewidth=1.5)
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

### PLOT 2 - policy for each period
for idx_method, method in enumerate(rm_list):
    for time_idx in env.spaces["t_space"][:-1][::-1]:
        hist2dim_pi = np.zeros([len(env.spaces["s_space"]), len(env.spaces["q_space"])])
        for s_idx, s_val in enumerate(env.spaces["s_space"]):
            for q_idx, q_val in enumerate(env.spaces["q_space"]):
                hist2dim_pi[len(env.spaces["s_space"])-s_idx-1, q_idx], _ = \
                        actor_critic.select_actions(T.Tensor([s_val]).to(actor_critic.device),
                                                    T.Tensor([q_val]).to(actor_critic.device),
                                                    T.tensor([time_idx]).to(actor_critic.device),
                                                    'best')

        # plot a 2D histogram of the terminal policy
        plt.imshow(hist2dim_pi,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(env.spaces["q_space"]),
                        np.max(env.spaces["q_space"]),
                        np.min(env.spaces["s_space"]),
                        np.max(env.spaces["s_space"])],
                aspect='auto',
                vmin=-env.params["max_u"],
                vmax=env.params["max_u"])
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Learned Policy; Time step:' + str(time_idx+1))
        plt.xlabel("Inventory")
        plt.ylabel("Price")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(repo + '/learnedpolicy_' + method + '_' + str(time_idx+1) + '.pdf', transparent=True)
        plt.clf()

# print progress
print('*** Testing phase completed! ***')
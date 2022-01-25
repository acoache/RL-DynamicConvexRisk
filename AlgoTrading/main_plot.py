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

costs = np.zeros((Nsimulations, envParams["Ndt"], len(rm_list))) # matrix to store all testing trajectories


for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the environment and risk measure objects
    env = TradingEnv(envParams)
    risk_measure = RiskMeasure(Type='mean')

    # create policy & value function objects
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
    s, q = env.reset(Nsimulations)
    
    for timestep in env.spaces["t_space"][:-1]:
        # simulate transitions according to the policy
        u, _ = actor_critic.select_actions(s, q, timestep*T.ones(Nsimulations), 'best')
        s, q, cost = env.step(s, q, u)

        # store costs
        costs[:,timestep,idx_method] = cost.detach().numpy()

    # get terminal reward
    costs[:,-1,idx_method] = env.get_final_cost(s, q).detach().numpy()

    ### PLOT - policy for each period
    for time_idx in env.spaces["t_space"][:-1][::-1]:
        # initialize 2D histogram
        hist2dim_pi = np.zeros([len(env.spaces["s_space"]), len(env.spaces["q_space"])])

        for s_idx, s_val in enumerate(env.spaces["s_space"]):
            for q_idx, q_val in enumerate(env.spaces["q_space"]):
                # best action according to the policy
                hist2dim_pi[len(env.spaces["s_space"])-s_idx-1, q_idx], _ = \
                        actor_critic.select_actions(T.Tensor([s_val]).to(actor_critic.device),
                                                    T.Tensor([q_val]).to(actor_critic.device),
                                                    T.tensor([time_idx]).to(actor_critic.device),
                                                    'best')

        # plot the 2D histogram
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
        
        plt.title('Learned Policy; Time step:' + str(time_idx+1))
        plt.xlabel("Inventory")
        plt.ylabel("Price")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(repo + '/learnedpolicy_' + method + '_' + str(time_idx+1) + '.pdf', transparent=True)
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
plt.xlabel("Terminal wealth")
plt.ylabel("Density")
plt.title("Distribution of the terminal wealth")

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

# print progress
print('*** Testing phase completed! ***')
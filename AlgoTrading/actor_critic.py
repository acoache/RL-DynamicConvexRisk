"""
# Policy gradient functions (actor-critic style algorithm)


"""
# numpy
import numpy as np
import numpy.matlib
# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# pytorch
import torch as T
import torch.optim as optim
from torch.distributions import Normal
# misc
import utils
from datetime import datetime
import pdb # use with set_trace() for the debugger

class ActorCriticPG():
    # constructor
    def __init__(self,
                    repo, # repository for files
                    method, # sub folder for files
                    env, # environment
                    policy, # ANN structure for the policy
                    V, # ANN structure for the value function
                    risk_measure, # risk measure
                    gamma=1): # discount factor

        assert (gamma > 0) and (gamma <= 1), "gamma needs to be in (0,1]"

        # assign objects to the actor_critic instance
        self.policy = policy # policy (ACTOR)
        self.V = V # value function (CRITIC)
        self.env = env # environment
        self.repo = repo # repository for files
        self.method = method # sub folder for files
        self.risk_measure = risk_measure # risk measure
        self.gamma = gamma # discount factor
        self.device = self.policy.device # PyTorch device
        
        # initialize loss objects
        self.loss_history_policy = [] # keep track of all losses for the policy
        self.loss_history_V = [] # keep track of all losses for the V

        # create lists and dictionaries for optimal policy
        self.states_list = []
        self.V_opt = {}
        self.best_actions = {}


    # mapping for actions: (-infty, infty) -> (lower, upper)
    def map_actions(self, x, lower, upper):
        return lower + (upper-lower) *  (1.0 / (1.0 + T.exp(-x)))


    # inverse mapping for actions: (lower, upper) -> (-infty, infty)
    def map_inv_actions(self, x, lower, upper):
        return T.log(x-lower+1e-7) - T.log(upper-x+1e-7)


    # select an action according to the policy ('best' or 'random')
    def select_actions(self, s_t, q_t, time_t, choose, seed=None):
        assert s_t.shape[0] == q_t.shape[0], "s and q must have same shape"
        assert q_t.shape[0] == time_t.shape[0], "q and time must have same shape"
        
        # freeze the set of random normal variables
        if seed is not None:
            T.manual_seed(seed)

        # observations as a formatted tensor
        obs_t = T.stack((s_t.clone(), q_t.clone(), time_t.clone()),-1)
        
        # obtain parameters of the distribution
        actions_param1, actions_param2 = self.policy(obs_t.clone())

        # create action distributions with a Normal distribution
        actions_dist = Normal(actions_param1, actions_param2)
     
        # get action from the policy
        if choose=='random':
            actions_sample = actions_dist.rsample()  # random sample from the Normal
        elif choose=='best':
            actions_sample = actions_param1  # mode of the Normal
        else:
            assert False, "Type of action selection is unknown ('random' or 'best')"
        
        # obtain lower and upper bounds for the actions
        min_u = T.maximum(T.tensor(-self.env.params["max_u"], device=self.device), -self.env.params["max_q"] - q_t)
        max_u = T.minimum(T.tensor(self.env.params["max_u"], device=self.device), self.env.params["max_q"] - q_t)
        
        # get actions from the mapping: (-infty, infty) -> (min_u, max_u)
        u_t = self.map_actions(actions_sample.squeeze(-1), min_u, max_u)

        # get log-probabilities of the action
        log_prob_t = actions_dist.log_prob(actions_sample.detach()).squeeze()
        
        # verification of any problem with log_prob
        if(T.isnan(log_prob_t).any() or T.isinf(log_prob_t).any()):
            assert False, "missing or infinite values in the gradients"
        
        return u_t, log_prob_t

    
    # simulate transitions from the policy
    def sim_transitions(self,
                        time_idx, # time index for the period
                        Ntrajectories=100, # number of trajectories
                        Mtransitions=100, # number of transitions
                        choose='random', # how to choose the actions
                        seed=None): # random seed
        
        # starting state
        s_0, q_0 = self.env.random_reset(Ntrajectories) # multiple random states

        # freeze the seed
        if seed is not None:
            T.manual_seed(seed)
        
        # initialize tables for all trajectories
        s = T.zeros((Ntrajectories, 2), \
                                dtype=T.float, requires_grad=False, device=self.device)
        q = T.zeros((Ntrajectories, 2), \
                                dtype=T.float, requires_grad=False, device=self.device)

        s_tp1 = T.zeros((Ntrajectories, Mtransitions), \
                                dtype=T.float, requires_grad=False, device=self.device)
        q_tp1 = T.zeros((Ntrajectories, Mtransitions), \
                                dtype=T.float, requires_grad=False, device=self.device)
        u_t = T.zeros((Ntrajectories, 1), \
                                dtype=T.float, requires_grad=False, device=self.device)
        log_prob_t = T.zeros((Ntrajectories, 1), \
                                dtype=T.float, requires_grad=False, device=self.device)
        cost_t = T.zeros((Ntrajectories, 1), \
                                dtype=T.float, requires_grad=False, device=self.device)
        
        # set initial values
        s[:,0] = s_0
        q[:,0] = q_0
        
        # get actions from the policy (outer)
        u_traj, _ = self.select_actions(s[:,0], q[:,0], time_idx, choose)

        # simulate transitions (outer)
        s[:,1], q[:,1], _ = self.env.step(s[:,0], q[:,0], u_traj)
        
        # get actions from the policy (inner)
        u_t, log_prob_t = self.select_actions(s[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                                  q[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                                  time_idx, choose)

        # simulate transitions (inner): multiple actions
        s_tp1, q_tp1, cost_t = self.env.step(s[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                              q[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                              u_t)

        # store (outer) trajectories in a dictionary
        trajs = {'s' : s, # starting and ending states -- price
                 'q' : q} # starting and ending states -- inventory

        # store (inner) transitions in a dictionary
        transitions = {'s_tp1' : s_tp1, # ending states from the actions -- price
                       'q_tp1' : q_tp1, # ending states from the actions -- inventory
                       'cost_t' : cost_t, # costs from the actions
                       'u_t' : u_t, # actions taken
                       'log_prob_t' : log_prob_t} # log-prob from the actions

        return trajs, transitions


    # simulate trajectories from the policy
    def sim_trajectories(self,
                        Ntrajectories=100, # number of trajectories
                        Mtransitions=100, # number of transitions
                        choose='random', # how to choose the actions
                        seed=None): # random seed
        
        # freeze the seed
        if seed is not None:
            T.manual_seed(seed)
        
        # initialize tables for all trajectories
        s = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        q = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)

        s_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        q_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        u_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        log_prob_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        cost_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
                
        # simulate N whole trajectories
        for t_idx in self.env.spaces["t_space"]:
            # starting state
            s[:,t_idx], q[:,t_idx] = self.env.random_reset(Ntrajectories) # multiple random states
            timestep[:,t_idx] = t_idx

            # get actions from the policy (inner)
            u_t[:,:,t_idx], log_prob_t[:,:,t_idx] = \
                            self.select_actions(s[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                q[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                timestep[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                choose)

            # simulate transitions (inner): multiple actions
            s_tp1[:,:,t_idx], q_tp1[:,:,t_idx], cost_t[:,:,t_idx] = \
                            self.env.step(s[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                            q[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                            u_t[:,:,t_idx])
            timestep_tp1[:,:,t_idx] = t_idx+1
        
        # store (outer) trajectories in a dictionary
        trajs = {'s' : s, # starting and ending states -- price
                 'q' : q, # starting and ending states -- inventory
                 'timestep' : timestep} # starting and ending states -- time index

        # store (inner) transitions in a dictionary
        transitions = {'s_tp1' : s_tp1, # ending states from the actions -- price
                       'q_tp1' : q_tp1, # ending states from the actions -- inventory
                       'timestep_tp1' : timestep_tp1, # ending states from the actions -- time index
                       'cost_t' : cost_t, # costs from the actions
                       'u_t' : u_t, # actions taken
                       'log_prob_t' : log_prob_t} # log-prob from the actions

        return trajs, transitions


    # estimate the value function for all time steps (critic)
    def estimate_V(self,
                    Ntrajectories, # number of trajectories
                    Mtransitions, # number of transitions
                    batch_size=50, # batch size for the update
                    Nepochs=100, # number of epochs
                    rng_seed=None): # random seed
        # print progress
        print('--Estimation of V--')
        batch_size = np.minimum(batch_size, Ntrajectories)

        # set V in training mode
        self.V.train()

        # generate full trajectories from policy
        trajs, transitions = self.sim_trajectories(Ntrajectories,
                                                    Mtransitions,
                                                    choose="random",
                                                    seed=rng_seed)
        
        for epoch in range(Nepochs):
            # zero grad
            self.V.zero_grad()

            # sample a batch of states at time t+1
            batch_idx = np.random.choice(Ntrajectories, size=batch_size, replace=False)
            s_batch = trajs["s"][batch_idx, 1:]
            q_batch = trajs["q"][batch_idx, 1:]
            time_batch = trajs["timestep"][batch_idx, 1:]
            
            # compute predicted values
            obs_t = T.stack((s_batch, q_batch, time_batch),-1).detach()
            v_pred = self.V(obs_t).squeeze()
            
            # compute target values
            v_target = T.zeros(v_pred.shape, requires_grad=False)

            # value function at the next time step
            obs_tp1 = T.stack((transitions["s_tp1"][batch_idx, :, 1:-1],
                               transitions["q_tp1"][batch_idx, :, 1:-1],
                               transitions["timestep_tp1"][batch_idx, :, 1:-1]),-1)
            v_tp1 = self.V(obs_tp1.clone()).squeeze()
            cost_t = transitions["cost_t"][batch_idx, :, 1:-1]
            
            # value function for last time step
            v_target[:, -1] = self.env.get_final_reward(s_batch[:,-1], q_batch[:,-1])
            
            # value function for other time steps
            v_target[:, :-1] = self.risk_measure.compute_risk(cost_t + v_tp1)
            
            # calculate the loss function
            v_loss = self.V.loss(v_target.detach(), v_pred).to(self.device)
            v_loss.backward()
            self.V.optimizer.step()
            self.loss_history_V.append(v_loss.detach().numpy())
        
            # print progress
            if epoch % 50 == 0 or epoch == Nepochs - 1:
                print('   Epoch = ',
                        str(epoch),
                        ', Loss: ',
                        str(np.round( np.mean(self.loss_history_V[-60:]) ,3)))

        # set V in evaluation mode
        self.V.eval()

    
    # update the policy according to a batch of trajectories (actor)
    def update_policy(self,
                        Ntrajectories, # number of trajectories
                        Mtransitions, # number of transitions
                        batch_size=50, # batch size for the update
                        Nepochs=100, # number of epochs
                        rng_seed=None): # random seed
        # print progress
        print('--Update of pi--')
        batch_size = np.minimum(batch_size, Ntrajectories)
        
        # set the policy in training mode
        self.policy.train()
        
        for epoch in range(Nepochs):
            # zero grad
            self.policy.zero_grad()
            
            # sample a batch of transitions
            trajs, transitions = self.sim_trajectories(batch_size,
                                                  Mtransitions,
                                                  choose='random',
                                                  seed=rng_seed)

            # value function of the next time step
            obs_tp1 = T.stack((transitions["s_tp1"][:, :, :-1].clone(),
                               transitions["q_tp1"][:, :, :-1].clone(),
                               transitions["timestep_tp1"][:, :, :-1].clone()),-1)
            V_tp1 = self.V(obs_tp1.clone()).squeeze()
            
            # combine both the cost and value function
            V_loss = self.risk_measure.get_V_loss(transitions["cost_t"][:, :, :-1].detach()+V_tp1.detach(),
                                                    transitions["log_prob_t"][:, :, :-1])

            # loss for each initial state
            grad_loss = V_loss 
            
            # average over all initial states and times
            loss = T.mean(grad_loss)
            
            # optimization step
            loss.to(self.device).backward()
            self.policy.optimizer.step()

            # store the loss
            self.loss_history_policy.append(loss.detach().numpy())

            # print progress
            if epoch % 50 == 0 or epoch == Nepochs - 1:
                print('   Epoch = ',
                      str(epoch) ,
                      ', Loss: ',
                      str(np.round( np.mean(self.loss_history_policy[-60:]) ,3)))

        # set the policy in evaluation mode
        self.policy.eval()


    # plot the strategy at any point in the algorithm
    def plot_current_policy(self, time_idx):
        # find the best actions
        hist2dim = np.zeros([len(self.env.spaces["s_space"]), len(self.env.spaces["q_space"])])
        for s_idx, s_val in enumerate(self.env.spaces["s_space"]):
            for q_idx, q_val in enumerate(self.env.spaces["q_space"]):
                hist2dim[len(self.env.spaces["s_space"])-s_idx-1, q_idx], _ = \
                        self.select_actions(T.Tensor([s_val]).to(self.device),
                                            T.Tensor([q_val]).to(self.device),
                                            T.tensor([time_idx]).to(self.device),
                                            'best')

        # plot the policy
        plt.imshow(hist2dim,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(self.env.spaces["q_space"]),
                        np.max(self.env.spaces["q_space"]),
                        np.min(self.env.spaces["s_space"]),
                        np.max(self.env.spaces["s_space"])],
                aspect='auto',
                vmin=-self.env.params["max_u"],
                vmax=self.env.params["max_u"])
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Best actions; Time step:' + str(time_idx+1))
        plt.xlabel("Inventory")
        plt.ylabel("Price")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.repo + '/' + self.method + 
                    '/time' + str(time_idx+1) +
                    '/best_actions_timestep' + str(time_idx+1) + 
                    '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) +
                    '.png', transparent=False)
        plt.clf()


    # plot the entire strategy at any point in the algorithm
    def plot_current_policies(self):
        for time_idx in self.env.spaces["t_space"][:-1][::-1]:
            self.plot_current_policy(time_idx)


    # plot the value function at any point in the algorithm
    def plot_current_V(self, time_idx):
        # find the best actions
        hist2dim = np.zeros([len(self.env.spaces["s_space"]), len(self.env.spaces["q_space"])])
        for s_idx, s_val in enumerate(self.env.spaces["s_space"]):
            for q_idx, q_val in enumerate(self.env.spaces["q_space"]):
                obs = T.stack((s_val*T.ones(1), q_val*T.ones(1), time_idx*T.ones(1)), -1)
                hist2dim[len(self.env.spaces["s_space"])-s_idx-1, q_idx] = self.V(obs)

        # plot the value function
        plt.imshow(hist2dim,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(self.env.spaces["q_space"]),
                        np.max(self.env.spaces["q_space"]),
                        np.min(self.env.spaces["s_space"]),
                        np.max(self.env.spaces["s_space"])],
                aspect='auto')
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Value function; Time step:' + str(time_idx+1))
        plt.xlabel("Inventory")
        plt.ylabel("Price")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.repo + '/' + self.method +
                    '/time' + str(time_idx+1) +
                    '/V_function_timestep' + str(time_idx+1) + 
                    '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) +
                    '.png', transparent=False)
        plt.clf()


    # plot the entire value function at any point in the algorithm
    def plot_current_Vs(self):
        for time_idx in self.env.spaces["t_space"][:-1][::-1]:
            self.plot_current_V(time_idx)


    ## functions to obtain the optimal policy and value function
    # obtain (discrete) state from a (continuous) observation
    def get_state(self, s, q):
        s_bin = np.digitize(s, self.env.spaces["s_space"])
        s_bin[s_bin == 0] = 1
        q_bin = np.digitize(q, self.env.spaces["q_space"])

        return s_bin, q_bin


    # give the set of valid actions
    def get_valid_actions(self, s, q):
        condition = np.abs(self.env.spaces["u_space"] + q) <= np.max(self.env.spaces["q_space"])
        valid_u = self.env.spaces["u_space"][ condition ]

        return valid_u


    # brute force method to compute the optimal policy / value function
    def optimal_policy(self, Nsims, plot=False):
        # initialize the state space
        for price in range(1, len(self.env.spaces["s_space"])+1):
            for inventory in range(1, len(self.env.spaces["q_space"])+1):
                self.states_list.append((price, inventory))

        # initialize the value function
        for time_idx in self.env.spaces["t_space"]:
            for state in self.states_list:
                self.V_opt[(time_idx, state)] = 0.0
                self.best_actions[(time_idx, state)] = 0.0
        
        # calculate the value function for the last time step
        for state in self.states_list:
            s = self.env.spaces["s_space"][state[0]-1]
            q = self.env.spaces["q_space"][state[1]-1]
            self.V_opt[(self.env.params["Ndt"]-1, state)] = self.env.get_final_reward_np(s, q)
        
        # obtain the value function recursively
        for time_idx in self.env.spaces["t_space"][:-1][::-1]:
            for state in self.states_list:
                
                # actual price and inventory
                s = self.env.spaces["s_space"][state[0]-1]
                q = self.env.spaces["q_space"][state[1]-1]
                
                # temporary value
                V_temp = np.inf
                action_temp = 0
                
                # obtain valid actions for that state
                valid_u = self.get_valid_actions(s, q)
                
                for action in valid_u:
                    # grab starting state and action
                    s_t = s * np.ones(Nsims)
                    q_t = q * np.ones(Nsims)
                    u_t = action * np.ones(Nsims)

                    # generate transitions
                    s_tp1, q_tp1, cost = self.env.step_np(s_t, q_t, u_t)
                    s_bin, q_bin = self.get_state(s_tp1, q_tp1)
                    
                    # value function of the next state
                    V_tp1 = np.asarray([self.V_opt[(time_idx+1, new_state)] for new_state in zip(s_bin, q_bin)])
                    
                    # compute the risk measure
                    temp = self.risk_measure.compute_risk_np(V_tp1)
                    new_V = temp + cost[0]
                    
                    # replace the optimal solution if improvement
                    if new_V < V_temp:
                        V_temp = new_V
                        action_temp = action
                
                # update the optimal value function and policy
                self.V_opt[(time_idx, state)] = V_temp
                self.best_actions[(time_idx, state)] = action_temp
            
            if plot:
                # compute the optimal value function and policy
                hist2dim_V = np.zeros([len(self.env.spaces["s_space"]), len(self.env.spaces["q_space"])])
                hist2dim_pi = np.zeros([len(self.env.spaces["s_space"]), len(self.env.spaces["q_space"])])
                for s_idx, s_val in enumerate(self.env.spaces["s_space"]):
                    for q_idx, q_val in enumerate(self.env.spaces["q_space"]):
                        obs = T.stack((s_val*T.ones(1), q_val*T.ones(1)), -1)
                        hist2dim_V[len(self.env.spaces["s_space"])-s_idx-1, q_idx] = \
                                                    self.V_opt[(time_idx+1, (s_idx+1, q_idx+1))]
                        hist2dim_pi[len(self.env.spaces["s_space"])-s_idx-1, q_idx] = \
                                                    self.best_actions[(time_idx, (s_idx+1, q_idx+1))]

                # plot a 2D histogram of the optimal value function
                plt.imshow(hist2dim_V,
                        interpolation='none',
                        cmap=utils.cmap,
                        extent=[np.min(self.env.spaces["q_space"]), 
                                np.max(self.env.spaces["q_space"]),
                                np.min(self.env.spaces["s_space"]),
                                np.max(self.env.spaces["s_space"])],
                        aspect='auto')
                plt.rcParams.update({'font.size': 16})
                plt.rc('axes', labelsize=20)
                plt.title('Value function (optimal); Time step:' + str(time_idx+1))
                plt.xlabel("Inventory")
                plt.ylabel("Price")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(self.repo + '/' + self.method +
                            '/optimalVF_time_idx' + str(time_idx+1) +
                            '.pdf', transparent=True)
                plt.clf()

                # plot a 2D histogram of the optimal policy
                plt.imshow(hist2dim_pi,
                        interpolation='none',
                        cmap=utils.cmap,
                        extent=[np.min(self.env.spaces["q_space"]), 
                                np.max(self.env.spaces["q_space"]),
                                np.min(self.env.spaces["s_space"]),
                                np.max(self.env.spaces["s_space"])],
                        aspect='auto',
                        vmin=-self.env.params["max_u"],
                        vmax=self.env.params["max_u"])
                plt.rcParams.update({'font.size': 16})
                plt.rc('axes', labelsize=20)
                plt.title('Best actions (optimal); Time step:' + str(time_idx+1))
                plt.xlabel("Inventory")
                plt.ylabel("Price")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(self.repo + '/' + self.method +
                            '/optimalACTIONS_time_idx' + str(time_idx+1) +
                            '.pdf', transparent=True)
                plt.clf()

            print('Optimal policy, time_idx = ', str(time_idx))
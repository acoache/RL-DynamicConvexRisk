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


    # select an action according to the policy ('best' or 'random')
    def select_actions(self, pos_t, time_t, choose, seed=None):
        assert pos_t.shape[0] == time_t.shape[0], "pos and time must have same shape"
        
        # freeze the set of random normal variables
        if seed is not None:
            T.manual_seed(seed)

        # observations as a formatted tensor
        obs_t = T.stack((pos_t.clone(), time_t.clone()),-1)
        
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
        
        # get actions
        u_t = actions_sample.squeeze(-1)

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
        pos_0 = self.env.random_reset(Ntrajectories) # multiple random states

        # freeze the seed
        if seed is not None:
            T.manual_seed(seed)
        
        # initialize tables for all trajectories
        pos = T.zeros((Ntrajectories, 2), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep = T.zeros((Ntrajectories, 2), \
                                dtype=T.float, requires_grad=False, device=self.device)

        pos_tp1 = T.zeros((Ntrajectories, Mtransitions), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep_tp1 = T.zeros((Ntrajectories, Mtransitions), \
                                dtype=T.float, requires_grad=False, device=self.device)
        u_t = T.zeros((Ntrajectories, 1), \
                                dtype=T.float, requires_grad=False, device=self.device)
        log_prob_t = T.zeros((Ntrajectories, 1), \
                                dtype=T.float, requires_grad=False, device=self.device)
        cost_t = T.zeros((Ntrajectories, 1), \
                                dtype=T.float, requires_grad=False, device=self.device)
        
        # set initial values
        pos[:,0] = pos_0
        timestep[:,0] = time_idx
        
        # get actions from the policy (outer)
        u_traj, _ = self.select_actions(pos[:,0], timestep[:,0], choose)

        # simulate transitions (outer)
        pos[:,1], timestep[:,1], _ = self.env.step(pos[:,0], timestep[:,0], u_traj)
        
        # get actions from the policy (inner)
        u_t, log_prob_t = self.select_actions(pos[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                                  timestep[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                                  choose)

        # simulate transitions (inner): multiple actions
        pos_tp1, timestep_tp1, cost_t = self.env.step(pos[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                              timestep[:,0].unsqueeze(-1).repeat(1,Mtransitions),
                                              u_t)

        # store (outer) trajectories in a dictionary
        trajs = {'pos' : pos, # starting and ending states -- position
                 'timestep' : timestep} # starting and ending states -- time

        # store (inner) transitions in a dictionary
        transitions = {'pos_tp1' : pos_tp1, # ending states from the actions -- position
                       'timestep_tp1' : timestep_tp1, # ending states from the actions -- time
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
        pos = T.zeros((Ntrajectories, self.env.params["T"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep = T.zeros((Ntrajectories, self.env.params["T"]), \
                                dtype=T.float, requires_grad=False, device=self.device)

        pos_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        u_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        log_prob_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        cost_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        
        # simulate N whole trajectories
        for t_idx in self.env.spaces["t_space"]:
            # starting state
            pos[:,t_idx] = self.env.random_reset(t_idx, Ntrajectories) # multiple random states
            timestep[:,t_idx] = t_idx

            # get actions from the policy (inner)
            u_t[:,:,t_idx], log_prob_t[:,:,t_idx] = \
                            self.select_actions(pos[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                timestep[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                choose)

            # simulate transitions (inner): multiple actions
            pos_tp1[:,:,t_idx], timestep_tp1[:,:,t_idx], cost_t[:,:,t_idx] = \
                            self.env.step(pos[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                            timestep[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                            u_t[:,:,t_idx])
            timestep_tp1[:,:,t_idx] = t_idx+1

        # store (outer) trajectories in a dictionary
        trajs = {'pos' : pos, # starting and ending states -- position
                 'timestep' : timestep} # starting and ending states -- time index

        # store (inner) transitions in a dictionary
        transitions = {'pos_tp1' : pos_tp1, # ending states from the actions -- position
                       'timestep_tp1' : timestep_tp1, # ending states from the actions -- time index
                       'cost_t' : cost_t, # costs from the actions
                       'u_t' : u_t, # actions taken
                       'log_prob_t' : log_prob_t} # log-prob from the actions

        return trajs, transitions


    # # simulate trajectories from the policy
    # def sim_trajectories(self,
    #                     Ntrajectories=100, # number of trajectories
    #                     Mtransitions=100, # number of transitions
    #                     choose='random', # how to choose the actions
    #                     seed=None): # random seed
        
    #     # freeze the seed
    #     if seed is not None:
    #         T.manual_seed(seed)
        
    #     # initialize tables for all trajectories
    #     pos = T.zeros((Ntrajectories, self.env.params["T"]), \
    #                             dtype=T.float, requires_grad=False, device=self.device)
    #     timestep = T.zeros((Ntrajectories, self.env.params["T"]), \
    #                             dtype=T.float, requires_grad=False, device=self.device)

    #     pos_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
    #                             dtype=T.float, requires_grad=False, device=self.device)
    #     timestep_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
    #                             dtype=T.float, requires_grad=False, device=self.device)
    #     u_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
    #                             dtype=T.float, requires_grad=False, device=self.device)
    #     log_prob_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
    #                             dtype=T.float, requires_grad=False, device=self.device)
    #     cost_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["T"]), \
    #                             dtype=T.float, requires_grad=False, device=self.device)
        
    #     # starting state
    #     pos[:,0] = self.env.reset(Ntrajectories) # multiple initial states
    #     timestep[:,0] = 0

    #     # simulate N whole trajectories
    #     for t_idx in self.env.spaces["t_space"]:
    #         if (t_idx != self.env.params["T"]-1):
    #             # get actions from the policy (outer)
    #             u_outer, _ = self.select_actions(pos[:,t_idx], timestep[:,t_idx], choose)

    #             # simulate transitions (outer): a single action
    #             pos[:,t_idx+1], timestep[:,t_idx+1], _ = self.env.step(pos[:,t_idx], timestep[:,t_idx], u_outer)

    #         # get actions from the policy (inner)
    #         u_t[:,:,t_idx], log_prob_t[:,:,t_idx] = \
    #                         self.select_actions(pos[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
    #                                             timestep[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
    #                                             choose)

    #         # simulate transitions (inner): multiple actions
    #         pos_tp1[:,:,t_idx], timestep_tp1[:,:,t_idx], cost_t[:,:,t_idx] = \
    #                         self.env.step(pos[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
    #                                         timestep[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
    #                                         u_t[:,:,t_idx])
    #         timestep_tp1[:,:,t_idx] = t_idx+1

    #     # store (outer) trajectories in a dictionary
    #     trajs = {'pos' : pos, # starting and ending states -- position
    #              'timestep' : timestep} # starting and ending states -- time index

    #     # store (inner) transitions in a dictionary
    #     transitions = {'pos_tp1' : pos_tp1, # ending states from the actions -- position
    #                    'timestep_tp1' : timestep_tp1, # ending states from the actions -- time index
    #                    'cost_t' : cost_t, # costs from the actions
    #                    'u_t' : u_t, # actions taken
    #                    'log_prob_t' : log_prob_t} # log-prob from the actions

    #     return trajs, transitions


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
            pos_batch = trajs["pos"][batch_idx, 1:]
            time_batch = trajs["timestep"][batch_idx, 1:]
            
            # compute predicted values
            obs_t = T.stack((pos_batch, time_batch),-1).detach()
            v_pred = self.V(obs_t).squeeze()
            
            # compute target values
            v_target = T.zeros(v_pred.shape, requires_grad=False)

            # value function at the next time step
            obs_tp1 = T.stack((transitions["pos_tp1"][batch_idx, :, 1:-1],
                               transitions["timestep_tp1"][batch_idx, :, 1:-1]),-1)
            v_tp1 = self.V(obs_tp1.clone()).squeeze()
            cost_t = transitions["cost_t"][batch_idx, :, 1:-1]
            
            # value function for last time step
            v_target[:, -1] = self.env.get_final_reward(pos_batch[:,-1])
            
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
                        str(np.round( np.mean(self.loss_history_V[-100:]) ,3)))
        
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
            obs_tp1 = T.stack((transitions["pos_tp1"][:, :, :-1].clone(),
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
                      str(np.round( np.mean(self.loss_history_policy[-100:]) ,3)))

        # set the policy in evaluation mode
        self.policy.eval()


    # plot the strategy at any point in the algorithm
    def plot_current_policy(self):
        hist2dim_pi = np.zeros([len(self.env.spaces["pos_space"]), len(self.env.spaces["t_space"])-1])
        for pos_idx, pos_val in enumerate(self.env.spaces["pos_space"]):
            for time_idx, time_val in enumerate(self.env.spaces["t_space"][:-1]):
                hist2dim_pi[len(self.env.spaces["pos_space"])-pos_idx-1, time_idx], _ = \
                        self.select_actions(T.Tensor([pos_val]).to(self.device),
                                            T.tensor([time_val]).to(self.device),
                                            'best')

        # plot the preferred path for the policy
        best_path = np.zeros(self.env.params["T"])
        for time_idx in self.env.spaces["t_space"][:-1]:
            action, _ = \
                    self.select_actions(T.Tensor([best_path[time_idx]]).to(self.device),
                                        T.tensor([time_idx]).to(self.device),
                                        'best')
            best_path[time_idx+1] = best_path[time_idx] + action

        # plot the policy
        plt.imshow(hist2dim_pi,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(self.env.spaces["t_space"]), 
                        np.max(self.env.spaces["t_space"]),
                        np.min(self.env.spaces["pos_space"]),
                        np.max(self.env.spaces["pos_space"])],
                aspect='auto',
                vmin=-self.env.params["max_u"],
                vmax=self.env.params["max_u"])
        plt.plot(np.arange(self.env.params["T"]),
            best_path,
            '-o',
            color=utils.mgreen,
            linewidth=1.5)
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Best actions')
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.repo + '/' + self.method +
                    '/evolution/best_actions' + 
                    '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) +
                    '.png', transparent=True)
        plt.clf()


    # plot the value function at any point in the algorithm
    def plot_current_V(self):
        hist2dim_V = np.zeros([len(self.env.spaces["pos_space"]), len(self.env.spaces["t_space"])-1])
        for pos_idx, pos_val in enumerate(self.env.spaces["pos_space"]):
            for time_idx, time_val in enumerate(self.env.spaces["t_space"][:-1]):
                obs = T.stack((pos_val*T.ones(1), time_val*T.ones(1)), -1)
                hist2dim_V[len(self.env.spaces["pos_space"])-pos_idx-1, time_idx] = self.V(obs)

        # plot the value function
        plt.imshow(hist2dim_V,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(self.env.spaces["t_space"]), 
                        np.max(self.env.spaces["t_space"]),
                        np.min(self.env.spaces["pos_space"]),
                        np.max(self.env.spaces["pos_space"])],
                aspect='auto')
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Value function')
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.repo + '/' + self.method +
                    '/evolution/V_function' + 
                    '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) +
                    '.png', transparent=True)
        plt.clf()


    ## functions to obtain the optimal policy and value function
    # obtain (discrete) state from a (continuous) observation
    def get_state(self, pos):
        pos_bin = np.digitize(pos, self.env.spaces["pos_space"])
        pos_bin[pos_bin == 0] = 1

        return pos_bin


    # brute force method to compute the optimal policy / value function
    def optimal_policy(self, Nsims, plot=False):
        if plot:
            hist2dim_V = np.zeros([len(self.env.spaces["pos_space"]), len(self.env.spaces["t_space"])-1])
            hist2dim_pi = np.zeros([len(self.env.spaces["pos_space"]), len(self.env.spaces["t_space"])-1])

        # initialize the state space
        for pos in range(1, len(self.env.spaces["pos_space"])+1):
            self.states_list.append((pos))
        
        # initialize the value function
        for time_idx in self.env.spaces["t_space"]:
            for state in self.states_list:
                self.V_opt[(time_idx, state)] = 0.0
                self.best_actions[(time_idx, state)] = 0.0
        
        # calculate the value function for the last time step
        for state in self.states_list:
            pos = self.env.spaces["pos_space"][state-1]
            self.V_opt[(self.env.params["T"]-1, state)] = self.env.get_final_reward_np(pos)
        
        # obtain the value function recursively
        for time_idx, time_val in enumerate(self.env.spaces["t_space"][:-1][::-1]):
            
            for state in self.states_list:
                
                # actual price and inventory
                pos = self.env.spaces["pos_space"][state-1]
                
                # temporary value
                V_temp = np.inf
                action_temp = 0
                
                for action in self.env.spaces["u_space"]:
                    # grab starting state and action
                    pos_t = pos * np.ones(Nsims)
                    time_t = time_val * np.ones(Nsims)
                    u_t = action * np.ones(Nsims)

                    # generate transitions
                    pos_tp1, time_tp1, cost = self.env.step_np(pos_t, time_t, u_t)
                    pos_bin = self.get_state(pos_tp1)
                    
                    # value function of the next state
                    V_tp1 = np.asarray([self.V_opt[(time_val+1, new_state)] for new_state in pos_bin])
                    
                    # compute the risk measure
                    new_V = self.risk_measure.compute_risk_np(V_tp1+cost)
                    
                    # replace the optimal solution if improvement
                    if new_V < V_temp:
                        V_temp = new_V
                        action_temp = action
                
                # update the optimal value function and policy
                self.V_opt[(time_val, state)] = V_temp
                self.best_actions[(time_val, state)] = action_temp
            
            if plot:
                # compute the optimal value function and policy
                for pos_idx, pos_val in enumerate(self.env.spaces["pos_space"]):
                    obs = T.stack((pos_val*T.ones(1), time_val*T.ones(1)), -1)
                    hist2dim_V[len(self.env.spaces["pos_space"])-pos_idx-1, self.env.params["T"]-time_idx-2] = \
                                self.V_opt[(time_val+1, (pos_idx+1))]
                    hist2dim_pi[len(self.env.spaces["pos_space"])-pos_idx-1, self.env.params["T"]-time_idx-2] = \
                                self.best_actions[(time_val, (pos_idx+1))]

            print('Optimal policy, time_idx = ', str(self.env.params["T"]-time_idx-1))
        
        if plot:
            # find the preferred path 
            best_path = np.zeros(self.env.params["T"])
            for time_idx in self.env.spaces["t_space"][:-1]:
                pos_bin = np.digitize(best_path[time_idx], self.env.spaces["pos_space"])
                pos_bin = pos_bin if pos_bin!=0 else 1
                action = self.best_actions[(time_idx, pos_bin)]
                best_path[time_idx+1] = best_path[time_idx] + action
            
            # plot a 2D histogram of the optimal value function
            plt.imshow(hist2dim_V,
                    interpolation='none',
                    cmap=utils.cmap,
                    extent=[np.min(self.env.spaces["t_space"]), 
                            np.max(self.env.spaces["t_space"]),
                            np.min(self.env.spaces["pos_space"]),
                            np.max(self.env.spaces["pos_space"])],
                    aspect='auto')
            plt.rcParams.update({'font.size': 16})
            plt.rc('axes', labelsize=20)
            plt.title('Value function (optimal)')
            plt.xlabel("Time")
            plt.ylabel("Position")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(self.repo + '/' + self.method +
                        '/optimalVF.pdf', transparent=True)
            plt.clf()

            # plot a 2D histogram of the optimal policy
            plt.imshow(hist2dim_pi,
                    interpolation='none',
                    cmap=utils.cmap,
                    extent=[np.min(self.env.spaces["t_space"]), 
                            np.max(self.env.spaces["t_space"]),
                            np.min(self.env.spaces["pos_space"]),
                            np.max(self.env.spaces["pos_space"])],
                    aspect='auto',
                    vmin=-self.env.params["max_u"],
                    vmax=self.env.params["max_u"])
            plt.plot(np.arange(self.env.params["T"]),
                best_path,
                '-o',
                color=utils.mgreen,
                linewidth=1.5)
            plt.rcParams.update({'font.size': 16})
            plt.rc('axes', labelsize=20)
            plt.title('Best actions (optimal)')
            plt.xlabel("Time")
            plt.ylabel("Position")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(self.repo + '/' + self.method +
                        '/optimalACTIONS.pdf', transparent=True)
            plt.clf()
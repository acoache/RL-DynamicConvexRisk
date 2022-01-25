"""
Hyperparameters
Initialization of all hyperparameters

"""

def cost_move_t(pos):
    return abs(pos)**2

def cost_move_T(pos):
    return abs(pos)**2

# initialize parameters for the environment and algorithm
def initParams():
    # name of the repository
    repo_name = 'CliffWalking_ex1'

    # parameters for the model
    envParams = {'T' : 9, # number of periods
              'cliff' : 1.0, # position of the cliff
              'C_cliff' : 100, # cost when falling into the cliff
              'C_time' : 1.0, # cost from a period to another
              'C_move' : cost_move_t, # cost from a movement
              'C_terminal' : cost_move_T, # terminal penalty on the position
              'max_u' : 4.0, # maximum of the mean of the Gaussian policy
              'sigma': 1.50} # standard deviation of the Gaussian policy

    # parameters for the algorithm
    algoParams = {'Ntrajectories' : 500, # number of generated trajectories
                    'Mtransitions' : 1000, # number of additional transitions for each state
                    'Nepochs' : 300, # number of epochs of the whole algorithm
                    'gamma' : 1.00, # discount factor
                    'Nepochs_V_init' : 1000, # number of epochs for the estimation of V during the first epoch
                    'Nepochs_V' : 75, # number of epochs for the estimation of V
                    'lr_V' : 1e-3, # learning rate of the neural net associated with V
                    'batch_V' : 200, # number of trajectories for each mini-batch in estimating V
                    'hidden_V' : 16, # number of hidden nodes in the neural net associated with V
                    'layers_V' : 4, # number of layers in the neural net associated with V
                    'Nepochs_pi' : 10, # number of epoch for the update of pi
                    'lr_pi' : 5e-4, # learning rate of the neural net associated with pi
                    'batch_pi' : 200, # number of trajectories for each mini-batch when updating pi
                    'hidden_pi' : 16, # number of hidden nodes in the neural net associated with pi
                    'layers_pi' : 3, # number of layers in the neural net associated with pi
                    'Nsims_optimal' : 1000, # number of simulations when using the brute force method
                    'seed' : None} # set seed for replication purposes

    return repo_name, envParams, algoParams

# print parameters for the environment and algorithm
def printParams(envParams, algoParams):
    print('*  T: ', envParams["T"],
            ' cliff: ', envParams["cliff"],
            ' C_cliff: ', envParams["C_cliff"],
            ' C_time: ', envParams["C_time"],
            ' max_u: ', envParams["max_u"],
            ' sigma: ', envParams["sigma"])
    print('*  Ntrajectories: ', algoParams["Ntrajectories"],
            ' Mtransitions: ', algoParams["Mtransitions"], 
            ' Nepochs: ', algoParams["Nepochs"],
            ' Nsims_optimal: ', algoParams["Nsims_optimal"])
    print('*  Nepochs_V_init: ', algoParams["Nepochs_V_init"],
            ' Nepochs_V: ', algoParams["Nepochs_V"],
            ' lr_V: ', algoParams["lr_V"], 
            ' batch_V: ', algoParams["batch_V"],
            ' hidden_V: ', algoParams["hidden_V"],
            ' layers_V: ', algoParams["layers_V"])
    print('*  Nepochs_pi: ', algoParams["Nepochs_pi"],
            ' lr_pi: ', algoParams["lr_pi"], 
            ' batch_pi: ', algoParams["batch_pi"],
            ' hidden_pi: ', algoParams["hidden_pi"],
            ' layers_pi: ', algoParams["layers_pi"])

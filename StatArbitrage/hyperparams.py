"""
Hyperparameters
Initialization of all hyperparameters

"""

# initialize parameters for the environment and algorithm
def initParams():
    # name of the repository
    repo_name = 'StatArbitrage_ex1'

    # parameters for the model
    envParams = {'kappa' : 2, # kappa of the OU process
              'sigma' : 0.2, # standard deviation of the OU process
              'theta' : 1, # mean-reversion level of the OU process
              'phi' : 0.005, # transaction costs
              'psi' : 0.5, # terminal penalty on the inventory
              'T' : 1, # trading horizon
              'Ndt' : 5+1, # number of periods
              'max_q' : 5, # maximum value for the inventory
              'max_u' : 2} # maximum value for the trades

    # parameters for the algorithm
    algoParams = {'Ntrajectories' : 500, # number of generated trajectories
                    'Mtransitions' : 500, # number of additional transitions for each state
                    'Nepochs' : 300, # number of epochs of the whole algorithm
                    'gamma' : 1.00, # discount factor
                    'Nepochs_V_init' : 500, # number of epochs for the estimation of V during the first epoch
                    'Nepochs_V' : 50, # number of epochs for the estimation of V
                    'lr_V' : 1e-3, # learning rate of the neural net associated with V
                    'batch_V' : 200, # number of trajectories for each mini-batch in estimating V
                    'hidden_V' : 32, # number of hidden nodes in the neural net associated with V
                    'layers_V' : 4, # number of layers in the neural net associated with V
                    'Nepochs_pi' : 10, # number of epoch for the update of pi
                    'lr_pi' : 1e-3, # learning rate of the neural net associated with pi
                    'batch_pi' : 200, # number of trajectories for each mini-batch when updating pi
                    'hidden_pi' : 32, # number of hidden nodes in the neural net associated with pi
                    'layers_pi' : 3, # number of layers in the neural net associated with pi
                    'Nsims_optimal' : 1000, # number of simulations when using the brute force method
                    'seed' : None} # set seed for replication purposes

    return repo_name, envParams, algoParams

# print parameters for the environment and algorithm
def printParams(envParams, algoParams):
    print('*  T: ', envParams["T"],
            ' Ndt: ', envParams["Ndt"],
            ' kappa: ', envParams["kappa"],
            ' sigma: ', envParams["sigma"],
            ' theta: ', envParams["theta"],
            ' phi: ', envParams["phi"],
            ' psi: ', envParams["psi"],
            ' max_q: ', envParams["max_q"],
            ' max_u: ', envParams["max_u"])
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

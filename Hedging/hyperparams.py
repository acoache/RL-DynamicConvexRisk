"""
Hyperparameters
Initialization of all hyperparameters

"""

# initialize parameters for the environment and algorithm
def initParams():
    # name of the repository
    repo_name = 'Hedging_ex1'

    # parameters for the model
    envParams = {'S0' : 10, # initial price
              'K' : 10, # strike price
              'v0' : 0.2**2, # initial volatility
              'kappa' : 9.0, # mean-reversion rate
              'theta' : 0.25**2, # mean-reversion level, long-run volatility
              'eta' : 1.0, # volatility of the volatility
              'T' : 1/12, # time length
              'rho' : -0.5, # correlation between brownian motions
              'mu' : 0.1, # drift
              'r' : 0.0, # interest rate of the bank account
              'B0' : 0.046, # initial wealth in the bank account
              'epsilon' : 0.005, # transaction costs
              'max_alpha' : 3.0, # maximal quantity of assets traded in one period
              'Ndt' : 31} # number of trading periods

    # parameters for the algorithm
    algoParams = {'Ntrajectories' : 500, # number of generated trajectories
                    'Mtransitions' : 500, # number of additional transitions for each state
                    'Nepochs' : 400, # number of epochs of the whole algorithm
                    'gamma' : 1.00, # discount factor
                    'Nepochs_V_init' : 1500, # number of epochs for the estimation of V during the first epoch
                    'Nepochs_V' : 300, # number of epochs for the estimation of V
                    'lr_V' : 5e-4, # learning rate of the neural net associated with V
                    'batch_V' : 200, # number of trajectories for each mini-batch in estimating V
                    'hidden_V' : 16, # number of hidden nodes in the neural net associated with V
                    'layers_V' : 4, # number of layers in the neural net associated with V
                    'Nepochs_pi' : 10, # number of epoch for the update of pi
                    'lr_pi' : 5e-4, # learning rate of the neural net associated with pi
                    'batch_pi' : 200, # number of trajectories for each mini-batch when updating pi
                    'hidden_pi' : 16, # number of hidden nodes in the neural net associated with pi
                    'layers_pi' : 3, # number of layers in the neural net associated with pi
                    'seed' : None} # set seed for replication purposes

    return repo_name, envParams, algoParams

# print parameters for the environment and algorithm
def printParams(envParams, algoParams):
    print('*  S0: ', envParams["S0"],
            ' K: ', envParams["K"],
            ' v0: ', envParams["v0"],
            ' kappa: ', envParams["kappa"],
            ' theta: ', envParams["theta"],
            ' eta: ', envParams["eta"],
            ' r: ', envParams["r"])
    print('*  T: ', envParams["T"],
            ' rho: ', envParams["rho"],
            ' mu: ', envParams["mu"],
            ' B0: ', envParams["B0"],
            ' epsilon: ', envParams["epsilon"],
            ' max_alpha: ', envParams["max_alpha"],
            ' Ndt: ', envParams["Ndt"])
    print('*  Ntrajectories: ', algoParams["Ntrajectories"],
            ' Mtransitions: ', algoParams["Mtransitions"], 
            ' Nepochs: ', algoParams["Nepochs"])
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

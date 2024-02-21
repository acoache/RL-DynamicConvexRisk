# Reinforcement Learning with Dynamic Convex Risk Measures

This Github repository regroups the Python code to run the actor-critic algorithm and replicate the experiments given in the paper [Reinforcement Learning with Dynamic Convex Risk Measures](https://doi.org/10.1111/mafi.12388) by [Anthony Coache](https://anthonycoache.ca/) and [Sebastian Jaimungal](http://sebastian.statistics.utoronto.ca/). There is one folder for each set of experiments, respectively the [statistical arbitrage](https://github.com/acoache/RL-DynamicConvexRisk/tree/main/StatArbitrage), [cliff walking](https://github.com/acoache/RL-DynamicConvexRisk/tree/main/CliffWalking) and [hedging](https://github.com/acoache/RL-DynamicConvexRisk/tree/main/Hedging) with friction examples. There is also a [Python notebook](https://github.com/acoache/RL-DynamicConvexRisk/blob/main/notebook.ipynb) to showcase how to use our code and replicate some of the experiments.


For further details on the algorithm and theoretical aspects of the problem, please refer to our [paper](https://doi.org/10.1111/mafi.12388).

Thank you for your interest in my research work. If you have any additional enquiries, please reach out to myself at anthony.coache@mail.utoronto.ca.

#### Authors

[Anthony Coache](https://anthonycoache.ca/) & [Sebastian Jaimungal](http://sebastian.statistics.utoronto.ca/)

*** 

All folders have the same structure, with the following files: 

* hyperparams.py
* main.py
* main_plot.py
* actor_critic.py 
* envs.py
* models.py
* risk_measure.py
* utils.py

***

### hyperparams.py

This file contains functions to initialize and print all hyperparameters, both for the environment and the actor-critic algorithm.

### main.py

This file contains the program to run the training phase. The first part concerns the importation of libraries and initialization of all parameters, either for the environment, neural networks or risk measure. Some notable parameters that need to be specified by the user in the hyperparams.py file are the numbers of epochs, learning rates, size of the neural networks and number of episodes/transitions among others. The next section is the training phase and its skeleton is given in the [paper](https://doi.org/10.1111/mafi.12388). It uses mostly functions from the actor_critic.py file. Finally, the models for the policy and value function are saved in a folder, along with diagnostic plots.

### main_plot.py

This file contains the program to run the testing phase. The first part concerns the importation of libraries and initialization of all parameters. Note that parameters must be identical to the ones used in main.py. The next section evaluates the policy found by the algorithm. It runs several simulations using the best behavior found by the actor-critic algorithm. Finally it outputs graphics to assess the performance of the procedure, such as the preferred action in any possible state and the estimated distribution of the cost when following the best policy.

### actor_critic.py

The whole algorithm is wrapped into a single class named ActorCriticPG, where input arguments specify which problem the agent faces. The user needs to give an environment, a (convex) risk measure, as well as two neural network structures that play the role of the value function and agent's policy. Each instance of that class has functions to select actions from the policy, whether at random or using the best behavior found thus far, and give the set of invalid actions. 
There is also a function to simulate (outer) episodes and (inner) transitions using the simulation upon simulation approach discussed in the paper. The update of the value function is wrapped in a function which takes as inputs the mini-batch size, number of epochs and characteristics of the value function neural network structure, such as the learning rate and the number of hidden nodes. Similarly, another function implements the update of the policy and takes as inputs the mini-batch size and number of epochs.

### envs.py

This file contains the environment class for the RL problem, as well as functions to interact with it. It has the [PyTorch](https://pytorch.org/) and [NumPy](https://numpy.org/) versions of the simulation engine. 

### models.py

Models are regrouped under this file with classes to build ANN structures using the [PyTorch](https://pytorch.org/) library.

### risk_measure.py

This file has the class that creates an instance of a risk measure, with functions to compute the risk and calculate its gradient. Risk measures currently implemented are the expectation, the conditional value-at-risk (CVaR), the mean-semideviation, a penalized version of the CVaR, and a linear combination of the mean and CVaR. More specifically, we have

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20E%28X%29%20%3D%20E%5BX%5D)

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctext%7BCVaR%7D_%7B%5Calpha%7D%28X%29%20%3D%20%5Csup_%7B%5Cxi%20%5Cin%20U%28P%29%7D%20E%5E%7B%5Cxi%7D%5BX%5D)

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctext%7BMSD%7D_%7B%5Ckappa%2Cr%7D%28X%29%20%3D%20E%5BX%5D%20&plus;%20%5Ckappa%20%5Cleft%28%20E%5B%28X-E%5BX%5D%29%5E%7Br%7D_%7B&plus;%7D%5D%20%5Cright%29%5E%7B1/r%7D)

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctext%7BCVaR-p%7D_%7B%5Calpha%2C%5Ckappa%7D%28X%29%20%3D%20%5Csup_%7B%5Cxi%20%5Cin%20U%28P%29%7D%20%5C%7B%20E%5E%7B%5Cxi%7D%5BX%5D%20-%20%5Ckappa%20E%5B%5Cxi%20%5Clog%20%5Cxi%5D%5C%7D)

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctext%7BE-CVaR%7D_%7B%5Calpha%2C%5Ckappa%7D%28X%29%20%3D%20%5Ckappa%20E%5BX%5D%20&plus;%20%281-%5Ckappa%29%20%5Ctext%7BCVaR%7D_%7B%5Calpha%7D%28X%29)

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20U%28P%29%20%3D%20%5Cleft%5C%7B%20%5Cxi%20%3A%20%5Csum_%7B%5Comega%7D%20%5Cxi%28%5Comega%29%20P%28%5Comega%29%20%3D%201%2C%20%5C%20%5Cxi%20%5Cin%20%5Cleft%5B0%2C%5Cfrac%7B1%7D%7B%5Calpha%7D%20%5Cright%5D%20%5Cright%5C%7D)

### utils.py

This file contains some useful functions and variables, such as a function to create new directories and colors for the visualizations.

***

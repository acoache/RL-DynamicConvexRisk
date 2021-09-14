# Reinforcement Learning with Dynamic Convex Risk Measures

This Github repository regroups the Python code to run the actor-critic algorithm and replicate the experiments given in the paper [Reinforcement Learning with Dynamic Convex Risk Measures](https://anthonycoache.ca/). There are two folders for the algorithmic trading and cliff walking problems respectively. Both folders have the same structure, with the following files: 

* main_trading.py (resp. main_cliff.py)
* main_plot.py
* actor_critic.py
* envs.py
* models.py
* risk_measure.py
* utils.py

For further details on the algorithm and theoretical aspects of the problem, please refer to our [paper](https://anthonycoache.ca/).

***

### main.py

This file contains the program to run the training phase. The first part concerns the importation of libraries and initialization of all parameters, either for the environment, neural networks or risk measure. Some notable parameters that need to be specified by the user are the numbers of epochs, learning rates and size of the neural networks among others.

The next section is the training phase and its skeleton is given in the [paper](https://anthonycoache.ca/). It uses mostly functions from the actor_critic.py file. Finally, the models for the policy and value function are saved in a folder, along with diagnostic plots.

### main_plot.py

This file contains the program to run the testing phase. The first past concerns the importation of libraries and initialization of all parameters. Note that parameters need to be identical to the ones used in main.py. The next section evaluates the policy found by the algorithm. It runs several simulations using the best behaviour found by the actor-critic algorithm. Finally it outputs graphics to assess the performance of the procedure, such as the preferred action in any possible state and the estimated distribution of the cost when following the best policy.

### actor_critic.py

The whole algorithm is wrapped into a single class named ActorCriticPG, where input arguments specify which problem the agent faces. The user needs to give an environment, two neural network structures that play the role of the value function and agent's policy, as well as a convex risk measure. Each instance of that class has functions to select actions from the policy, whether at random or using the best behavior found so far, and give the set of invalid actions. There is also a function to simulate transitions for a batch of states. The update of the value function is wrapped in a function which takes as inputs the mini-batch size, number of epochs and characteristics of the value function neural network structure, such as the learning rate and the number of hidden nodes. Similarly, another function implements the update of the policy and takes as inputs the mini-batch size and number of epochs.

### envs.py

This file contains the environment class for the RL problem, as well as functions to interact with it. It has the [Pytorch](https://pytorch.org/) and [NumPy](https://numpy.org/) versions of the simulation engine. 

### risk_measure.py

This file has the class that creates an instance of a risk measure, with functions to compute the risk and calculate its gradient. The different risk measures implemented are the expectation, the conditional value-at-risk (CVaR), the mean-semideviation, a penalized version of the CVaR, and a linear combination of the mean and CVaR.

### utils.py

This file contains some useful functions and variables, such as a function to create new directories and colors for the visualizations.

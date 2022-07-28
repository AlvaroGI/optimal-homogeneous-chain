# optimal-homogeneous-chain-with-cutoffs

Optimal entanglement distribution policies in homogeneous repeater chains with cutoffs.
For a review of the mathemathical theory behind the code in this repo and analysis of the results, please see our paper ([arXiv:2207.06533](https://arxiv.org/abs/2207.06533)).

We formulate the problem of finding optimal policies as a Markov decision process (MDP), and we solve it using value and policy iteration (both methods are equivalent in terms of results). Next, we explain how to use our code to find optimal policies with value and policy iteration, and to perform a Monte Carlo simulation of a specific policy. We also provide the code used to derive the results shown in our paper.

The main variables employed in this project are the following:

 - `n`: number of nodes.

 - `p`: probability of successful entanglement generation.

 - `p_s`: probability of successful swap.

 - `cutoff`: cutoff time.


---


## Value iteration
In value iteration, we decided to first build the model (i.e., calculate all the transition probabilities in the Bellman equations) and then apply value iteration to solve it.
 
 - `valueiter_build.py`: generates the model, while applying the state bunching technique (see our paper for more details). An example of how to use this script can be found in `run_valueiter_build.sh`, which takes as input specific values for `n` and `cutoff` (note that we do not need the values of `p` and `p_s` to build the model).

 - `valueiter_solve.py`: finds an optimal policy using value iteration. This can only be used after generating the model with `valueiter_build.py`. An example of how to use this script can be found in `run_valueiter_solve.sh`.

 - `main.py`: contains some functions necessary for our value iteration algorithm.


---


## Policy iteration
In our implementation of policy iteration, we calculate transition probabilities when needed (using an environment), instead of building the model at the beginning, as we did in our implementation of value iteration. This way, we avoid storing large data files containing the model, at the cost of some extra computation time.

 - `policyiter_solve.py`: finds an optimal policy using policy iteration. An example of how to use this script can be found in `run_policyiter.sh`.

 - `main.py`: contains the main functions used to find optimal policies. The function `policy_iteration()` can also be called directly to find an optimal policy.

 - `policy.py` and `environment.py` are part of the policy iteration algorithm. These files are necessary to run `main.policy_iteration()`.


---


## Monte Carlo simulation
We coded a Monte Carlo simulation to validate the expected delivery times obtained from the MDP. This simulation can also be used to estimate the probability distribution of the delivery time.

 - `main.py`: contains `simulate_environment()`, which can run a simulation of the repeater chain. Examples can be found in `experiment_delivery-time-distribution.ipynb`.


---


## Data analysis and results
The following files can be used to generate the results that appear in our paper. The data for these plots has to be generated independently using the scripts explained in the previous sections. Our data will be made public soon.

 - `experiment_expected-delivery-times.ipynb`: expected delivery time of an optimal policy for different parameter combinations.

 - `experiment_scaling-states.ipynb`: scaling of the number of states in the MDP.

 - `experiment_optimal-actions-analysis.ipynb`: number of states in which the policies decide to perform all swaps or none.

 - `experiment_swapasap-vs-nested.ipynb`: we show that a nested policy can be significantly better than swap-asap when swaps are probabilistic.

 - `experiment_optimal-vs-swapasap.ipynb`: expected delivery time of optimal policies versus that of the swap-asap policy.

 - `experiment_delivery-time-distribution.ipynb`: delivery time distribution of an optimal policy (using Monte Carlo simulations).





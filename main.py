from environment import Environment
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from pathlib import Path
import pickle
from policy import Agent
import random
import time
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# ------------------------------------------------------------------ #
# ---------------------   POLICY ITERATION  ------------------------ #
# ------------------------------------------------------------------ #

def policy_iteration(n, p, p_s, cutoff, tolerance=1e-7, 
                     tolerance_stability=1e-1,
                     progress=True, savedata=True):
    '''Find the optimal policy in a homogeneous repeater chain with
        a policy iteration algorithm.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) the algorithm stops when the policy
                            is stable and the maximum difference between
                            a value and the updated value is smaller than
                            this tolerance. This impacts the error on the
                            delivery time.
            · tolerance_stability:  (float) tolerance used until the policy
                                    is stable. If tolerance_stability is
                                    larger than tolerance, the algorithm
                                    converges faster. If it is too large,
                                    the algorithm may not converge.
            · progress: (bool) if True, prints progress.
            · savedata: (bool) if True, saves the outputs in a file.
        ---Outputs---
            · v0_evol:  (list of lists) each list contains the evolution
                        of the value of the empty state (no entangled links)
                        over one policy evaluation step. The final value
                        is v0_evol[-1][-1].
            · state_info: (list of dicts) each dictionary corresponds to
                          a different state of the MDP, and it contains
                          the following keys:
                - state: matrix representation of the state. Element ij
                         contains the age of the entangled link shared
                         between nodes i and j. If the link does not exist,
                         the age is inf. In policy iteration, we consider
                         states just before swaps are performed.
                - action_space: list of valid actions. Each action is a
                                set of nodes that must perform swaps.
                - policy: probability that each element in the action space
                          is chosen by the optimal policy (since there is
                          always a deterministic optimal policy in a finite
                          MDP, this will be a one-hot vector).
                - value: value of the state when following said policy. The
                         expected delivery time from this state can be
                         calculated as -(value+1).
            · exe_time: (int) execution time of the algorithm in seconds.'''

    ## Check input values are valid ##
    assert type(n)==int, 'n must be an integer.'
    assert n>2, 'n must be larger than two.'
    assert p>=0 and p<=1, 'p must be between zero and one.'
    assert p_s>=0 and p_s<=1, 'p_s must be between zero and one.'
    assert type(cutoff)==int, 'cutoff must be an integer.'
    assert cutoff>0, 'cutoff must be larger than zero.'

    ## Initialize ##
    start_time = time.time()
    s0 = np.full(shape=(n,n), fill_value=np.infty) # Initial state
    action_space = [None]
    agent = Agent(s0=s0, max=cutoff)
    environment = Environment(p, cutoff, p_s=p_s)
    v0_evol = [] # Evolution of all the values
    policy_is_stable = False
    tol = tolerance_stability

    ## Policy iteration ##
    while True:

        ### Policy evaluation step ###
        v0 = []
        error = -np.inf
        while abs(error) > tol:
            idx = 0
            error = -np.inf
            while idx < len(agent.state_list):
                state = agent.get(idx, 'state')
                value = agent.get(idx, 'value')
                policy = agent.get(idx, 'policy')
                action_space = agent.get(idx, 'action_space')

                # Check for terminal states
                if environment.check_e2e_link(state):
                    idx+=1
                    continue

                # Loop over possible actions
                v = 0
                for action, action_prob in zip(action_space, policy):
                    states_nxt, probs_nxt, actions_nxt = environment.step(state, action)
                    assert len(states_nxt) == len(probs_nxt)
                    assert len(states_nxt) == len(actions_nxt)
                    for s, P, a in zip(states_nxt, probs_nxt, actions_nxt):
                        if P == 0.0: continue
                        s_idx = agent.observe(s, a)
                        s_value = agent.get(s_idx, 'value')
                        v += action_prob*P*(-1 + s_value)

                # Calculate error
                error = max(error, abs(value-v))

                # Update value
                agent.update(idx, v, 'value')
                idx +=1

            v0.append(agent.get(0,'value'))
            if progress:
                print('Policy eval.: error = %.2e > %.2e = tolerance'%(error,tol), end='\r')
        v0_evol.append(v0)

        ### Stop if policy has converged ###
        if policy_is_stable and policy_was_stable:
            break

        ### Policy improvement step ###
        idx = 0
        policy_was_stable = policy_is_stable
        policy_is_stable = True
        tol = tolerance
        while idx < len(agent.state_list):
            state = agent.get(idx, 'state')
            policy = agent.get(idx, 'policy')
            action_space = agent.get(idx, 'action_space')

            if (len(action_space) == 1) or environment.check_e2e_link(state):
                idx +=1
                continue

            q_values = [0.0 for action in action_space]
            for k, action in enumerate(action_space):
                states_nxt, probs_nxt, _ = environment.step(state, action)
                for s, P in zip(states_nxt, probs_nxt):
                    if P == 0.0: continue
                    s_idx = agent.observe(s)
                    s_value = agent.get(s_idx, 'value')
                    q_values[k] += P*(-1+s_value)

            best_action = np.argmax(q_values)
            new_policy = [0.0 for action in action_space]
            new_policy[best_action] = 1.0

            # Have we changed the policy at this state?
            if policy[best_action] < 1.0:
                policy_is_stable = False
                tol = tolerance_stability
                agent.update(idx, new_policy, 'policy')

            idx +=1
            if progress:
                print('Policy improvement: state %d/%d'%(idx,len(agent.state_list))+' '*40,
                                                         end='\r')

    if progress:
        print(' '*80, end='\r') # Clear line

    end_time = time.time()
    exe_time = end_time-start_time

    if savedata:
        save_policyiter_data(n, p, p_s, cutoff, tolerance, v0_evol,
                             agent.state_info, exe_time)

    return v0_evol, agent.state_info, exe_time

def check_policyiter_data(n, p, p_s, cutoff, tolerance):
    '''If policy iteration has been run and saved for this set of
        parameters, return True. Otherwise, return False.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.'''
    filename = 'data_policyiter/n%s_p%.3f_ps%.3f_tc%s_tol%s'%(n,p,p_s,cutoff,tolerance)
    if Path(filename).exists():
        return True
    else:
        return False

def save_policyiter_data(n, p, p_s, cutoff, tolerance, v0_evol, 
                         state_info, exe_time):
    '''Save policy iteration data for this set of parameters.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · v0_evol:  (list of lists) each list contains the evolution
                        of the value of the empty state (no entangled links)
                        over one policy evaluation step. The final value
                        is v0_evol[-1][-1].
            · state_info: (list of dicts) each dictionary corresponds to
                          a different state of the MDP, and it contains
                          the following keys:
                - state: matrix representation of the state. Element ij
                         contains the age of the entangled link shared
                         between nodes i and j. If the link does not exist,
                         the age is inf.
                - action_space: set of possible actions. Each action is a
                                set of nodes that must perform swaps.
                - policy: probability that each element in the action space
                          is chosen by the optimal policy (since there is
                          always a deterministic optimal policy in a finite
                          MDP, this will be a one-hot vector).
                - value: value of the state when following said policy. The
                         expected delivery time from this state can be
                         calculated as -(value+1).
            · exe_time: (int) execution time of the algorithm in seconds.'''
    # Create data directory if needed
    try:
        os.mkdir('data_policyiter')
    except FileExistsError:
        pass

    # Save data
    filename = 'data_policyiter/n%s_p%.3f_ps%.3f_tc%s_tol%s'%(n,p,p_s,cutoff,tolerance)
    data = {'v0_evol': v0_evol, 'state_info': state_info, 'exe_time': exe_time}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_policyiter_data(n, p, p_s, cutoff, tolerance):
    '''Load optimal policy data obtained via policy iteration.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
        ---Outputs---
            · v0_evol:  (list of lists) each list contains the evolution
                        of the value of the empty state (no entangled links)
                        over one policy evaluation step. The final value
                        is v0_evol[-1][-1].
            · state_info: (list of dicts) each dictionary corresponds to
                          a different state of the MDP, and it contains
                          the following keys:
                - state: matrix representation of the state. Element ij
                         contains the age of the entangled link shared
                         between nodes i and j. If the link does not exist,
                         the age is inf.
                - action_space: set of possible actions. Each action is a
                                set of nodes that must perform swaps.
                - policy: probability that each element in the action space
                          is chosen by the optimal policy (since there is
                          always a deterministic optimal policy in a finite
                          MDP, this will be a one-hot vector).
                - value: value of the state when following said policy. The
                         expected delivery time from this state can be
                         calculated as -(value+1).
            · exe_time: (int) execution time of the algorithm in seconds.'''
    filename = 'data_policyiter/n%s_p%.3f_ps%.3f_tc%s_tol%s'%(n,p,p_s,cutoff,tolerance)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['v0_evol'], data['state_info'], data['exe_time']

def classify_states_policyiter(state_info, print_policy=False):
    ''' ---Inputs---
        · state_info: (list of dicts) each dictionary corresponds to
                      a different state of the MDP, and it contains
                      the following keys:
            - state: matrix representation of the state. Element ij
                     contains the age of the entangled link shared
                     between nodes i and j. If the link does not exist,
                     the age is inf.
            - action_space: set of possible actions. Each action is a
                            set of nodes that must perform swaps.
            - policy: probability that each element in the action space
                      is chosen by the optimal policy (since there is
                      always a deterministic optimal policy in a finite
                      MDP, this will be a one-hot vector).
            - value: value of the state when following said policy. The
                     expected delivery time from this state can be
                     calculated as -(value+1).
        ---Outputs---
        · states_total: (int) total number of states.
        · states_nonterminal:   (int) total number of non-terminal or
                                non-absorbing states.
        · states_decision:  (int) number of non-terminal states in which
                            more than one action is valid (i.e., at least
                            one swap can be performed).
        · states_nowait:    (int) number of states in which the action chosen
                            was to swap something instead of waiting.
        · states_swapall:   (int) number of states in which all possible swaps
                            are performed.'''
    if print_policy:
        print('%25s  %26s\n'%('State','Action'))
    env = Environment(0.5,2) # These values are not relevant
    states_total = len(state_info)
    states_nonterminal = 0
    states_decision = 0
    states_noswap = 0
    states_nowait = 0
    states_swapall = 0
    for e in state_info:
        # Build state vector
        state = e['state']
        #if not env.check_e2e_path(state): # Only print non-terminal states
        if not env.check_e2e_link(state): # Only print non-terminal states
            states_nonterminal += 1
            if len(env.generate_action_space(state))>1: # Only states with more than 1 possible action
                states_decision += 1
                # Find the action that should be performed with probability 1
                k = 0
                while e['policy'][k] < 1:
                    k += 1
                a = e['action_space'][k]
                    
                # If action is to perform a swap:
                if a:
                    # The policy is acting differently from a waiting policy
                    # only if there is no e2e path
                    #if not env.check_e2e_path(state):
                    #    states_nowait += 1
                    states_nowait += 1
                    # If all swaps are performed
                    if e['policy'][-1] == 1:
                        states_swapall += 1
                    if print_policy:
                        s_id = []
                        s = []
                        for i in range(0, len(state)):
                            for j in range(i+1, min(i-1,0)+len(state)):
                                s_id += ['%d-%d'%(i,j)]
                                s += [state[i][j]]
                        print('[%s]'%', '.join(map(str, s_id)))
                        print('%10s  %6s\n'%(s,a))
                else:
                    states_noswap += 1

    if print_policy:
        print('Total number states: %d'%states_total)
        print('Non-terminal states: %d'%states_nonterminal)
        print('Decision states: %d'%states_decision)
        print('No-swap states: %d'%states_noswap)
        print('No-wait states: %d'%states_nowait)
        print('Swap-all states: %d'%states_swapall)

    return states_total, states_nonterminal, states_decision, states_noswap, states_nowait, states_swapall


# ------------------------------------------------------------------ #
# ----------------------   VALUE ITERATION  ------------------------ #
# ------------------------------------------------------------------ #

def check_valueiter_model(n, cutoff):
    '''If the model for value iteration has been generated and saved for
        this set of parameters, return True. Otherwise, return False.
        Note that the model only depends on the number of nodes
        and the cutoff.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.'''
    filename = 'data_valueiter/model/MDPmodel_n%s_tc%s_equations.pkl'%(n,cutoff)
    if Path(filename).exists():
        return True
    else:
        return False

def check_valueiter_data(n, p, p_s, cutoff, tolerance, randomseed=2):
    '''If value iteration has been run and saved for this set of
        parameters, return True. Otherwise, return False.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · randomseed:   (int) random seed.'''
    filename = ('data_valueiter/solution/MDPsol_n%s_p%.3f_ps%.3f'%(n,p,p_s)+
                '_tc%s_tol%s_randomseed%s'%(cutoff,tolerance,randomseed))
    if Path(filename).exists():
        return True
    else:
        return False

def load_valueiter_data(n, p, p_s, cutoff, tolerance, randomseed=2):
    '''Load optimal policy data obtained via policy iteration.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · randomseed:   (int) random seed.
        ---Outputs---
            · data: (dict) it contains the following keys:
                - state_labels: (array of tuples) element i of this array
                                contains the labels [j,k] of the two nodes
                                j and k that are connected by the link in
                                position i of the state array.
                - states:   (array of arrays) each array is a different state.
                            Element i of a state array corresponds to the
                            age of the link connecting nodes state_labels[i].
                - policy:   (list of lists) each list contains the action
                            performed at the corresponding state.
                - values:   (list) the i-th element is the expected delivery
                            time when starting from state i.'''
    filename = ('data_valueiter/solution/MDPsol_n%s_p%.3f_ps%.3f'%(n,p,p_s)+
                '_tc%s_tol%s_randomseed%s'%(cutoff,tolerance,randomseed))
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

def classify_states_valueiter(n, p, p_s, t_cut, tolerance, randomseed):
    ''' ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · randomseed:   (int) random seed.
        ---Outputs---
        · states_total: (int) total number of states.
        · states_decision:  (int) number of non-terminal states in which
                            more than one action is valid (i.e., at least
                            one swap can be performed).
        · states_noswap:    (int) number of states in which the action chosen
                            was to not perform any swap.
        · states_swapasap:  (int) number of states in which all possible swaps
                            are performed.'''
    data = load_valueiter_data(n, p, p_s, t_cut, tolerance, randomseed)

    # Auxiliary variables
    states_total = 0
    states_decision = 0
    states_noswap = 0
    states_swapasap = 0

    # Loop over states that exist at the beginning of a time slot
    for state_idx, state_policy in enumerate(data['policy']):

        # Loop over states that result after entanglement generation
        for afterGEN_policy in state_policy:
            afterGEN_policy = afterGEN_policy.tolist()

            # The state produced after entanglement gen is unique
            states_total += 1

            if len(afterGEN_policy) == 1:
                # No decisions to be made
                continue

            # Decision to be made
            states_decision += 1
            for trans_idx, transition in enumerate(afterGEN_policy):
                if transition[2] == 1:
                    # Pick transition that has probability 1
                    break

            # The first element is to not swap, so there is only 1 outcome
            assert len(afterGEN_policy[0][3]) == 1
            afterGEN_prob = afterGEN_policy[0][3][0]

            if trans_idx == 0:
                # The first transition is to not swap
                states_noswap += 1
            elif trans_idx == len(afterGEN_policy)-1:
                # The last transition is to swap all (i.e., swap-asap)
                states_swapasap += 1
            else:
                pass

    return states_total, states_decision, states_noswap, states_swapasap


# ------------------------------------------------------------------ #
# ---------------------   SWAP-ASAP POLICY  ------------------------ #
# ------------------------------------------------------------------ #

def policy_eval_swapasap(n, p, p_s, cutoff, tolerance=1e-5,
                           progress=True, savedata=False, nested=False):
    '''Find the expected delivery time of the swap-asap policy in a
        homogeneous repeater chain with the policy evaluation step from
        our policy iteration algorithm.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) the algorithm stops when the policy
                            is stable and the maximum difference between
                            a value and the updated value is smaller than
                            this tolerance. This impacts the error on the
                            delivery time.
            · progress: (bool) if True, prints progress.
            · savedata: (bool) if True, saves the outputs in a file.
            · nested:   (bool) if True and n=5, the swap-asap policy is
                        modified as follows. In a full state (in which
                        every pair of neighbors shares an entangled link),
                        instead of performing swaps at nodes two, three,
                        and four, only perform swaps at nodes two and four.
        ---Outputs---
            · v0_evol:  (list of lists) each list contains the evolution
                        of the value of the empty state (no entangled links)
                        over the policy evaluation. The final value
                        is v0_evol[-1][-1].
            · state_info: (list of dicts) each dictionary corresponds to
                          a different state of the MDP, and it contains
                          the following keys:
                - state: matrix representation of the state. Element ij
                         contains the age of the entangled link shared
                         between nodes i and j. If the link does not exist,
                         the age is inf. In policy iteration, we consider
                         states just before swaps are performed.
                - action_space: list of valid actions. Each action is a
                                set of nodes that must perform swaps.
                - policy: probability that each element in the action space
                          is chosen by the swap-asap policy, i.e., this will
                          be a one-hot vector with the one in the last position.
                - value: value of the state when following the swap-asap policy.
                         The expected delivery time from this state can be
                         calculated as -(value+1).
            · exe_time: (int) execution time of the algorithm in seconds.'''

    # If savedata is True, we will also save checkpoints
    checkpoints = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    if nested:
        assert n==5, 'Nested scheme only implemented for n=5 nodes'
        if savedata==True:
            raise ValueError('nested+savedata not implemented')

    ## Initialize ##
    start_time = time.time()
    s0 = np.full(shape=(n,n), fill_value=np.infty)
    action_space = [None]
    agent = Agent(s0=s0, max=cutoff)
    environment = Environment(p, cutoff, p_s=p_s)
    v0_evol = []

    ### Policy evaluation  ###
    v0 = []
    error = -np.inf
    while abs(error) > tolerance:
        idx = 0
        error = -np.inf
        while idx < len(agent.state_list):
            state = agent.get(idx, 'state')
            value = agent.get(idx, 'value')
            policy = agent.get(idx, 'policy')
            action_space = agent.get(idx, 'action_space')

            # Check for terminal states
            if environment.check_e2e_link(state):
                idx+=1
                continue

            # Set swap-asap
            policy_swapasap = [0.0 for action in policy]
            policy_swapasap[-1] = 1.0
            if nested:
                if len(policy_swapasap)==8: # In a 5-node chain, there are 8 possible
                                            # actions only if all the small links exist
                    policy_swapasap = [0.0 for action in policy]
                    for act_idx, act in enumerate(action_space):
                        if act == [1,3]:
                            break
                    policy_swapasap[act_idx] = 1.0

            agent.update(idx, policy_swapasap, 'policy')

            # Loop over possible actions
            v = 0
            for action, action_prob in zip(action_space, policy):
                states_nxt, probs_nxt, actions_nxt = environment.step(state, action)
                assert len(states_nxt) == len(probs_nxt)
                assert len(states_nxt) == len(actions_nxt)
                for s, P, a in zip(states_nxt, probs_nxt, actions_nxt):
                    if P == 0.0: continue
                    s_idx = agent.observe(s, a)
                    s_value = agent.get(s_idx, 'value')
                    v += action_prob*P*(-1 + s_value)

            # Calculate error
            error = max(error, abs(value-v))

            # Update value
            agent.update(idx, v, 'value')
            idx +=1

        v0.append(agent.get(0,'value'))
        if progress:
            print('Policy eval.: error = %.2e > %.2e = tolerance'%(error,tolerance), end='\r')

        # Save checkpoints
        if savedata:
            for checkpoint_tol in checkpoints:
                if error<checkpoint_tol:
                    end_time = time.time()
                    exe_time = end_time-start_time
                    v0_evol_cp = v0_evol
                    v0_evol_cp.append(v0)
                    save_swapasap_data(n, p, p_s, cutoff, checkpoint_tol, v0_evol_cp, agent.state_info, exe_time)
                    checkpoints = checkpoints[1:]

    v0_evol.append(v0)

    if progress:
        print(' '*80, end='\r') # Clear line

    end_time = time.time()
    exe_time = end_time-start_time

    if savedata:
        save_swapasap_data(n, p, p_s, cutoff, tolerance, v0_evol, agent.state_info, exe_time)

    return v0_evol, agent.state_info, exe_time

def check_swapasap_data(n, p, p_s, cutoff, tolerance):
    '''If policy evaluation on swap-asap has been run and saved for
        this set of parameters, return True. Otherwise, return False.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.'''
    filename = 'data_swapasap/swapasap_n%s_p%.3f_ps%.3f_tc%s_tol%s'%(n,p,p_s,cutoff,tolerance)
    if Path(filename).exists():
        return True
    else:
        return False

def save_swapasap_data(n, p, p_s, cutoff, tolerance,
                       v0_evol, state_info, exe_time):
    '''Save policy iteration data for this set of parameters.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · v0_evol:  (list of lists) each list contains the evolution
                        of the value of the empty state (no entangled links)
                        over one policy evaluation step. The final value
                        is v0_evol[-1][-1].
            · state_info: (list of dicts) each dictionary corresponds to
                          a different state of the MDP, and it contains
                          the following keys:
                - state: matrix representation of the state. Element ij
                         contains the age of the entangled link shared
                         between nodes i and j. If the link does not exist,
                         the age is inf.
                - action_space: set of possible actions. Each action is a
                                set of nodes that must perform swaps.
                - policy: probability that each element in the action space
                          is chosen by the optimal policy (since there is
                          always a deterministic optimal policy in a finite
                          MDP, this will be a one-hot vector).
                - value: value of the state when following said policy. The
                         expected delivery time from this state can be
                         calculated as -(value+1).
            · exe_time: (int) execution time of the algorithm in seconds.'''
    # Create data directory if needed
    try:
        os.mkdir('data_swapasap')
    except FileExistsError:
        pass

    # Save data
    filename = 'data_swapasap/swapasap_n%s_p%.3f_ps%.3f_tc%s_tol%s'%(n,p,p_s,cutoff,tolerance)
    data = {'v0_evol': v0_evol, 'state_info': state_info, 'exe_time': exe_time}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_swapasap_data(n, p, p_s, cutoff, tolerance):
    '''Load optimal policy data obtained via policy iteration.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
        ---Outputs---
            · v0_evol:  (list of lists) each list contains the evolution
                        of the value of the empty state (no entangled links)
                        over one policy evaluation step. The final value
                        is v0_evol[-1][-1].
            · state_info: (list of dicts) each dictionary corresponds to
                          a different state of the MDP, and it contains
                          the following keys:
                - state: matrix representation of the state. Element ij
                         contains the age of the entangled link shared
                         between nodes i and j. If the link does not exist,
                         the age is inf.
                - action_space: set of possible actions. Each action is a
                                set of nodes that must perform swaps.
                - policy: probability that each element in the action space
                          is chosen by the optimal policy (since there is
                          always a deterministic optimal policy in a finite
                          MDP, this will be a one-hot vector).
                - value: value of the state when following said policy. The
                         expected delivery time from this state can be
                         calculated as -(value+1).
            · exe_time: (int) execution time of the algorithm in seconds.'''
    filename = 'data_swapasap/swapasap_n%s_p%.3f_ps%.3f_tc%s_tol%s'%(n,p,p_s,cutoff,tolerance)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['v0_evol'], data['state_info'], data['exe_time']


# ------------------------------------------------------------------ #
# -------------------   MONTE CARLO SIMULATION  -------------------- #
# ------------------------------------------------------------------ #

def simulate_environment(policy, n, p, p_s, cutoff, N_samples=100,
                         randomseed=2, tolerance=1e-7,
                         progress_bar='notebook', savedata=False, test=False):
    '''Performs a Monte Carlo simulation of the environment used
        for policy iteration. For each sample, the repeater chain
        evolves until end-to-end entanglement is produced, and the
        delivery time is logged.
        ---Inputs---
            · policy:   (str) policy to be followed:
                - 'optimal':    optimal policy for this set of parameters.
                                The optimal policy must have been found
                                and saved using policy iteration in advance.
                - 'swap-asap':  perform swaps as soon as possible.
                - 'wait':   only perform swaps when all pairs of neighbors
                            share an entangled link.
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · N_samples:    (int) number of samples.
            · randomseed:   (int) random seed.
            · tolerance:    (float) tolerance used in policy iteration.
                            This input is only necessary if policy='optimal'.
            · progress_bar: (str) if 'notebook', shows a notebook-style progress
                            bar. Otherwise, shows a terminal-style bar.
            · savedata: (bool) if True, saves the outputs in a file.
        ---Outputs---
            · data_T:   (dict) dictionary with the following keys:
                - avg:  (float) average delivery time over all samples.
                - std:  (float) standard deviation in the delivery time.
                - N_samples:    (int) number of samples.
                - hist: (array) histogram of delivery times (this is more
                        efficient than storing the result of each sample
                        separately).
                - bins: (array) array containing the limits of each bin
                        in the histogram.'''

    np.random.seed(randomseed)
    random.seed(randomseed)
    if progress_bar == 'notebook':
        tqdm_ = tqdm_notebook
    elif progress_bar == 'plain':
        tqdm_ = tqdm
    else:
        raise ValueError('Invalid value for progress_bar.')

    if policy == 'optimal':
        nowait_states, nowait_actions = read_policy(n,p,p_s,cutoff,tolerance)

    rng = np.random.RandomState(0)

    T_vec = [] # Array that stores delivery times of all samples

    for j in tqdm_(range(N_samples), leave=False):
        # Initialize state
        done = False
        state = np.full(shape=(n,n), fill_value=np.infty)
        environment = Environment(p, cutoff, p_s=p_s)
        action_space = environment.generate_action_space(state)

        # Run policy
        time = 0
        while not done:
            # Choose next action
            if policy == 'swap-asap':
                m = 0
                action = action_space[0]
                for a in action_space:
                    if len(a) > m:
                        m = len(a)
                        action = a
            elif policy == 'optimal':
                action = []
                for s in nowait_states:
                    s = np.array(s)
                    if (state == s).all():
                        action = nowait_actions[nowait_states.index(state.tolist())]
                        break
            elif policy == 'wait':
                if environment.check_e2e_path(state):
                    m = 0
                    action = action_space[0]
                    for a in action_space:
                        if len(a) > m:
                            m = len(a)
                            action = a
                else:
                    action = []
            else:
                raise ValueError('Unknown policy')

            # Perform action
            s_out, p_out, a_out = environment.step(state, action)
            idx_next = rng.choice( len(s_out), p=p_out )
            state = s_out[idx_next]
            action_space = a_out[idx_next]
                
            done = environment.check_e2e_link(state)
            time+=1
        T_vec.append(time-1)

    # Store data as a histogram (for efficiency)
    bins_T = np.arange(-0.5, max(T_vec)+1.5)
    T_hist = np.histogram(T_vec, bins=bins_T)
    T_hist = T_hist[0]/sum(T_hist[0])
    T_values = np.arange(0, len(T_hist))
    data_T = {'avg': np.sum(np.multiply(T_values, T_hist)),
              'std': np.std(T_vec),
              'N_samples': N_samples,
              'bins': bins_T,
              'hist': T_hist}
    
    if savedata:
        # Create data directory if needed
        try:
            os.mkdir('data_sim')
        except FileExistsError:
            pass

        # Save data
        filename = ('data_sim/%s_n%d_p%.3f_ps%.3f_tc%s'%(policy,n,p,p_s,cutoff)
                    +'_samples%d_randseed%s'%(N_samples,randomseed))
        if policy=='optimal':
            filename += '_tol%s.pickle'%tolerance
        else:
            filename += '.pickle'

        with open(filename, 'wb') as handle:
            pickle.dump(data_T, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_T

def read_policy(n, p, p_s, cutoff, tolerance, print_statistics=False):
    '''For a specific combination of parameters, provides a list with
        the states in which the action of an optimal policy is to perform
        at least one swap, and the corresponding action. The optimal policy
        must be found and saved in advance using policy_iteration().
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used in policy iteration.
                            This input is only necessary if policy='optimal'.
            · print_statistics: (str) if True, prints the total number of
                                states, number non-terminal states,
                                number of states with more than one
                                possible action, and number of states where
                                the optimal policy decides to perform at
                                least one swap.
        ---Outputs---
            · nowait_states:    (list) states in which the optimal policy
                                decides to perform at least one swap.
            · nowait_actions:   (list) actions chosen by an optimal policy
                                in the states from nowait_states.'''

    # Load data (if exists)
    if check_policyiter_data(n, p, p_s, cutoff, tolerance):
        v0_evol, state_info, _ = load_policyiter_data(n, p, p_s, cutoff, tolerance)
    else:
        raise ValueError('Data not found for n=%d, p=%.3f,'%(n,p)+
                         'p_s=%.3f, cutoff=%d, tol=%e'%(p_s,cutoff,tolerance))

    env = Environment(p, cutoff) # These values are not relevant
    nowait_states = [] # States where the optimal action is to perform some swap(s)
    nowait_actions = []
    c_nonterminal = 0
    c_decision = 0
    for e in state_info:
        # Build state vector
        state = e['state']
        if not env.check_e2e_link(state): # Only consider non-terminal states
            c_nonterminal += 1
            if len(env.generate_action_space(state))>1: # States with more than 1 action
                # Find the action that should be performed with probability 1
                c_decision += 1
                k = 0

                try:
                    while e['policy'][k] < 1:
                        k += 1
                    action = e['action_space'][k]
                except: # XXX: This happens in states with e2e path in old data (p_s=1),
                        # since those were terminal states in the old implementation
                    action = e['action_space'][-1]

                # Save state only if action is to perform a swap
                if action:
                    nowait_states += [state.tolist()]
                    nowait_actions += [action]

    if print_statistics:
        print('Total number states: %d'%len(state_info))
        print('Non-terminal states: %d'%c_nonterminal)
        print('Decision states: %d'%c_decision)
        print('No-wait states: %d'%len(nowait_states))

    return nowait_states, nowait_actions

def check_sim_data(policy, n, p, p_s, cutoff, N_samples,
                          randomseed, tolerance=1e-7):
    '''If simulate_environment() has been run and saved for
        this set of parameters, return True. Otherwise, return False.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · N_samples:    (int) number of samples.
            · randomseed:   (int) random seed.
            · tolerance:    (float) tolerance used in policy iteration.
                            This input is only necessary if policy='optimal'.'''
    filename = ('data_sim/%s_n%s_p%.3f_ps%.3f_tc%s'%(policy,n,p,p_s,cutoff)
                    +'_samples%d_randseed%s'%(N_samples,randomseed))
    if policy=='optimal':
        filename += '_tol%s.pickle'%tolerance
    else:
        filename += '.pickle'

    if Path(filename).exists():
        return True
    else:
        return False

def load_sim_data(policy, n, p, p_s, cutoff, N_samples,
                  randomseed, tolerance=1e-7):
    '''Load simulation data obtained via simulate_environment().
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · N_samples:    (int) number of samples.
            · randomseed:   (int) random seed.
            · tolerance:    (float) tolerance used in policy iteration.
                            This input is only necessary if policy='optimal'.'''
    filename = ('data_sim/%s_n%s_p%.3f_ps%.3f_tc%s'%(policy,n,p,p_s,cutoff)
                    +'_samples%d_randseed%s'%(N_samples,randomseed))
    if policy=='optimal':
        filename += '_tol%s.pickle'%tolerance
    else:
        filename += '.pickle'

    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    return data


# ------------------------------------------------------------------ #
# -------------------   DATA ANALYSIS / PLOTS  --------------------- #
# ------------------------------------------------------------------ #

def gatherdata_waiting_swapasap(n, p, p_s, cutoff, tolerance, randomseed,
                               varying_params, varying_arrays, solver,
                               progress_bar='notebook'):
    '''Generate a matrix where each element contains the number of states
        in which an optimal policy decides to not perform any swap for a
        different combination of parameters. Do the same for states in which
        an optimal policy decides to perform all swaps (i.e., swap-asap).
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · randomseed:   (int) random seed used in value iteration.
            · varying_params:   (tuple of str) we scan over these parameters.
                                Should be 'n', 'p', or 'cutoff'.
            · varying_arrays:   (tuple of arrays) values of the varying_params
                                that will be analyzed.
            · solver:   (str) algorithm used to solve the MPD: 'valueiter'
                        or 'policyiter'.
            · progress_bar: (str) if 'notebook', shows a notebook-style progress
                            bar. Otherwise, shows a terminal-style bar.
        ---Outputs---
            · surf_waiting: (array) matrix containing the number of states in
                            which an optimal policy decides to not perform any
                            swap, for each combination of values of varying_params.
            · surf_swapasap:    (array) matrix containing the number of states in
                                which an optimal policy decides to perform all swaps,
                                for each combination of values of varying_params.'''
    surf_waiting = np.zeros(( len(varying_arrays[0]), len(varying_arrays[1]) ))
    surf_swapasap = np.zeros(( len(varying_arrays[0]), len(varying_arrays[1]) ))

    if progress_bar == 'notebook':
        tqdm_ = tqdm_notebook
    else:
        tqdm_ = tqdm

    i = 0
    for varying_value0 in tqdm_(varying_arrays[0],
                                '--Scanning %s...'%varying_params[0], leave=False):
        if varying_params[0] == 'n':
            n = varying_value0
        elif varying_params[0] == 'p':
            p = varying_value0
        elif varying_params[0] == 'cutoff':
            cutoff = varying_value0
        else:
            raise ValueError('varying_params[0] has an invalid value.')
        
        j = 0
        for varying_value1 in tqdm_(varying_arrays[1],
                                    '----Scanning %s...'%varying_params[1], leave=False):
            if varying_params[1] == 'n':
                n = varying_value1
            elif varying_params[1] == 'p':
                p = varying_value1
            elif varying_params[1] == 'cutoff':
                cutoff = varying_value1
            else:
                raise ValueError('varying_params[1] has an invalid value.')

            if solver == 'valueiter':
                _, states_decision, states_noswap, states_swapasap = classify_states_valueiter(n, p, p_s, cutoff, tolerance, randomseed)
            elif solver == 'policyiter':
                _, state_info, _ = load_policyiter_data(n, p, p_s, cutoff, tolerance)
                _, _, states_decision, states_noswap, _, states_swapasap = classify_states_policyiter(state_info, print_policy=False)
            else:
                raise ValueError('Unknown solver %s'%solver)

            surf_waiting[i,j] = states_noswap / states_decision * 100
            surf_swapasap[i,j] = states_swapasap / states_decision * 100

            j += 1
        i += 1

    return surf_waiting, surf_swapasap

def makefig_waiting_swapasap(n, p, p_s, cutoff, tolerance, randomseed,
                            varying_params, varying_arrays,
                            surf_waiting, surf_swapasap, solver, savefig=False):
    '''Generate figure with the data gathered in gatherdata_waiting_swapasap().
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · randomseed:   (int) random seed used in value iteration.
            · varying_params:   (tuple of str) we scan over these parameters.
                                Should be 'n', 'p', or 'cutoff'.
            · varying_arrays:   (tuple of arrays) values of the varying_params
                                that will be analyzed.
            · surf_waiting: (array) matrix containing the number of states in
                            which an optimal policy decides to not perform any
                            swap, for each combination of values of varying_params.
            · surf_swapasap:    (array) matrix containing the number of states in
                                which an optimal policy decides to perform all swaps,
                                for each combination of values of varying_params.
            · solver:   (str) algorithm used to solve the MPD: 'valueiter'
                        or 'policyiter'.
            · savefig:  (bool) if True, saves figure.'''
    fontsizes = 9
    fontsizes_ticks = fontsizes-1
    x_cm = 9
    y_cm = 4

    if varying_params[0] == 'n':
            xlab = '$n$'
    elif varying_params[0] == 'p':
        xlab = '$p$'
    elif varying_params[0] == 'cutoff':
        xlab = '$t_\mathrm{cut}$'
    else:
        raise ValueError('varying_param has an invalid value.')

    if varying_params[1] == 'n':
        ylab = '$n$'
    elif varying_params[1] == 'p':
        ylab = '$p$'
    elif varying_params[1] == 'cutoff':
        ylab = '$t_\mathrm{cut}$'
    else:
        raise ValueError('varying_param has an invalid value.')

    dx = (varying_arrays[0][1]-varying_arrays[0][0])/2
    dy = (varying_arrays[1][1]-varying_arrays[1][0])/2

    for surf_idx, surf in enumerate([surf_waiting, surf_swapasap]):
        fig, ax = plt.subplots(figsize=(x_cm/2.54, y_cm/2.54))

        surfmax = np.max(np.abs(surf))
        surfmin = np.min(np.abs(surf))
        cbar_min = np.floor(surfmin/10)*10
        cbar_max = np.ceil(surfmax/10)*10
        if surf_idx == 0:
            cbar_min = 0
            cbar_max = 50
        cbar_mid = cbar_min+(cbar_max-cbar_min)/2
        num_levels = int(cbar_max-cbar_min)

        cmap = plt.cm.get_cmap('Blues', num_levels)
        cont = ax.imshow(np.flip(surf.T, 0), cmap=cmap,
                          extent=[varying_arrays[0][0]-dx, varying_arrays[0][-1]+dx,
                                  varying_arrays[1][0]-dy, varying_arrays[1][-1]+dy],
                          vmin=cbar_min, vmax=cbar_max)
        ax.set_aspect(aspect="auto")

        ## Plot specs ##
        ax.set_xticks(np.arange(0.3,0.99,0.1))
        # Minor x-tick frequency
        x_minor_intervals = 5 # Number of minor intervals between two major ticks               
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(x_minor_intervals))
        ax.set_yticks(varying_arrays[1][::2])
        # Minor y-tick frequency
        y_minor_intervals = 2 # Number of minor intervals between two major ticks               
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(y_minor_intervals))
        plt.xlabel(r'$%s$'%xlab, fontsize=fontsizes)
        plt.ylabel(r'$%s$'%ylab, fontsize=fontsizes)
        ax.tick_params(labelsize=fontsizes_ticks)

        ## Colorbar ##
        fontsize_cbar_label = fontsizes
        if surf_idx==0:
            cbar_label = 'Waiting states'
        else:
            cbar_label = 'Swap-asap states'
        cbar = fig.colorbar(cont, ax=ax, aspect=10)#, format='%d')
        cbar.set_label(cbar_label,
                       fontsize=fontsize_cbar_label)
        if surfmax == surfmin:
            cbar.set_ticks([0,surfmax])
            cbar.ax.set_yticklabels([r'${:.0f}\%$'.format(0),
                                     r'${:.0f}\%$'.format(surfmax)])
        else:
            cbar.set_ticks([cbar_min,cbar_mid,cbar_max])
            cbar.ax.set_yticklabels([r'${:.0f}\%$'.format(cbar_min),
                                     r'${:.0f}\%$'.format(cbar_mid),
                                     r'${:.0f}\%$'.format(cbar_max)])
        cbar.ax.tick_params(labelsize=fontsizes_ticks)

        if savefig:
            filename = 'MANUSCRIPT_figs/states'
            if surf_idx==0:
                filename += '_waiting_'
            elif surf_idx==1:
                filename += '_swap-asap_'
            if solver == 'valueiter' or solver == 'policyiter':
                filename += solver
            else:
                raise ValueError('Unknown solver')
            filename += '_n%d_ps%.3f_tolerance%s_randomseed%s.pdf'%(n,
                                             p_s,tolerance,randomseed)

            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def gatherdata_optimal_vs_swapasap(n, p, p_s, cutoff, tolerance, randomseed,
                                   varying_params, varying_arrays, solver,
                                   progress_bar='notebook'):
    '''Generate a matrix where each element contains the relative difference
        in expected delivery time between an optimal policy and the swap-asap
        policy for a different combination of parameters.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · randomseed:   (int) random seed used in value iteration.
            · varying_params:   (tuple of str) we scan over these parameters.
                                Should be 'n', 'p', or 'cutoff'.
            · varying_arrays:   (tuple of arrays) values of the varying_params
                                that will be analyzed.
            · solver:   (str) algorithm used to solve the MPD: 'valueiter'
                        or 'policyiter'.
            · progress_bar: (str) if 'notebook', shows a notebook-style progress
                            bar. Otherwise, shows a terminal-style bar.'''
    surf = np.zeros(( len(varying_arrays[0]), len(varying_arrays[1]) ))

    if progress_bar == 'notebook':
        tqdm_ = tqdm_notebook
    else:
        tqdm_ = tqdm

    i = 0
    for varying_value0 in tqdm_(varying_arrays[0],
                                '--Scanning %s...'%varying_params[0], leave=False):
        if varying_params[0] == 'n':
            n = varying_value0
        elif varying_params[0] == 'p':
            p = varying_value0
        elif varying_params[0] == 'cutoff':
            cutoff = varying_value0
        else:
            raise ValueError('varying_params[0] has an invalid value.')
        
        j = 0
        for varying_value1 in tqdm_(varying_arrays[1],
                                    '----Scanning %s...'%varying_params[1], leave=False):
            if varying_params[1] == 'n':
                n = varying_value1
            elif varying_params[1] == 'p':
                p = varying_value1
            elif varying_params[1] == 'cutoff':
                cutoff = varying_value1
            else:
                raise ValueError('varying_params[1] has an invalid value.')

            # Optimal policy
            if solver == 'valueiter':
                try:
                    data = load_valueiter_data(n, p, p_s, cutoff, tolerance, randomseed)
                    T_opt = data['values'][0]
                except: # If data does not exist
                    T_opt = None
            elif solver == 'policyiter':
                try:
                    _, state_info, _ = load_policyiter_data(n, p, p_s, cutoff, tolerance)
                    T_opt = -(state_info[0]['value']+1)
                except: # If data does not exist
                    T_opt = None
            else:
                raise ValueError('Unknown solver')
            
            # Swap-asap policy
            try:
                _, state_info, _ = load_swapasap_data(n, p, p_s, cutoff, tolerance)
                T_swap = -(state_info[0]['value']+1)
            except: # If data does not exist
                T_swap = None

            if T_swap!=None and T_opt!=None:
                surf[i,j] = (T_swap-T_opt) / T_opt * 100
            else:
                surf[i,j] = None

            j += 1
        i += 1

    return surf

def makefig_optimal_vs_swapasap(n, p, p_s, cutoff, tolerance, randomseed,
                                varying_params, varying_arrays, surf, solver,
                                cbar_max=15, annotate00=True, annotate11=False,
                                savefig=False):
    '''Generate figure with the data gathered in gatherdata_optimal_vs_swapasap().
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
            · tolerance:    (float) tolerance used as stopping condition
                            in the policy iteration algorithm.
            · randomseed:   (int) random seed used in value iteration.
            · varying_params:   (tuple of str) we scan over these parameters.
                                Should be 'n', 'p', or 'cutoff'.
            · varying_arrays:   (tuple of arrays) values of the varying_params
                                that will be analyzed.
            · surf: (array) output matrix from gatherdata_optimal_vs_swapasap().
                    Each element contains the relative difference in expected
                    delivery time between an optimal policy and the swap-asap
                    policy for a different combination of parameters.
            · solver:   (str) algorithm used to solve the MPD: 'valueiter'
                        or 'policyiter'.
            · cbar_max: (int) maximum value of the colorbar.
            · annotate00:   (bool) if True, annotates the value in the cell
                            corresponding to the smallest values of the varying
                            parameters.
            · annotate11:   (bool) if True, annotates the value in the cell
                            corresponding to the largest values of the varying
                            parameters.
            · savefig:  (bool) if True, saves figure.'''
    fontsizes = 9
    fontsizes_ticks = fontsizes-1
    x_cm = 9
    y_cm = 4

    if varying_params[0] == 'n':
            xlab = '$n$'
    elif varying_params[0] == 'p':
        xlab = '$p$'
    elif varying_params[0] == 'cutoff':
        xlab = '$t_\mathrm{cut}$'
    else:
        raise ValueError('varying_param has an invalid value.')

    if varying_params[1] == 'n':
        ylab = '$n$'
    elif varying_params[1] == 'p':
        ylab = '$p$'
    elif varying_params[1] == 'cutoff':
        ylab = '$t_\mathrm{cut}$'
    else:
        raise ValueError('varying_param has an invalid value.')

    dx = (varying_arrays[0][1]-varying_arrays[0][0])/2
    dy = (varying_arrays[1][1]-varying_arrays[1][0])/2

    fig, ax = plt.subplots(figsize=(x_cm/2.54, y_cm/2.54))

    surfmax = np.max(np.abs(surf))
    surfmin = np.min(np.abs(surf))
    cbar_min = 0
    cbar_mid = cbar_min+(cbar_max-cbar_min)/2

    cmap = plt.cm.get_cmap('Blues')
    cont = ax.imshow(np.flip(surf.T, 0), cmap=cmap,
                      extent=[varying_arrays[0][0]-dx, varying_arrays[0][-1]+dx,
                              varying_arrays[1][0]-dy, varying_arrays[1][-1]+dy],
                      vmin=cbar_min, vmax=cbar_max)
    ax.set_aspect(aspect="auto")

    # Plot specs #

    ax.set_xticks(np.arange(0.3,0.91,0.2))
    # Minor x-tick frequency
    x_minor_intervals = 2 # Number of minor intervals between two major ticks               
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(x_minor_intervals))
    
    ax.set_yticks(varying_arrays[1][::2])
    # Minor y-tick frequency
    y_minor_intervals = 2 # Number of minor intervals between two major ticks               
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(y_minor_intervals))
    
    plt.xlabel(r'$%s$'%xlab, fontsize=fontsizes)
    plt.ylabel(r'$%s$'%ylab, fontsize=fontsizes)
    ax.tick_params(labelsize=fontsizes_ticks)

    # Colorbar #

    fontsize_cbar_label = fontsizes
    cbar_label = r'$\frac{T_\mathrm{swap}-T_\mathrm{opt}}{T_\mathrm{opt}}$'

    cbar = fig.colorbar(cont, ax=ax, aspect=10)#, format='%d')
    cbar.set_label(cbar_label, fontsize=fontsize_cbar_label)
    if surfmax == surfmin:
        cbar.set_ticks([0,surfmax])
        cbar.ax.set_yticklabels([r'${:.0f}\%$'.format(0),
                                 r'${:.0f}\%$'.format(surfmax)])
    else:
        cbar.set_ticks([cbar_min,cbar_mid,cbar_max])
        cbar.ax.set_yticklabels([r'${:.0f}\%$'.format(cbar_min),
                                 r'${:.0f}\%$'.format(cbar_mid),
                                 r'${:.0f}\%$'.format(cbar_max)])
    cbar.ax.tick_params(labelsize=fontsizes_ticks)

    # Annotated colormap
    for iix, ii in enumerate(varying_arrays[0]):
        for jjx, jj in enumerate(varying_arrays[1]):
            if ((annotate00 and jjx==0 and iix==0) or
                (annotate11 and iix==len(varying_arrays[0])-1
                 and jjx==len(varying_arrays[1])-1)):
                if surf.T[jjx,iix] > cbar_max/2:
                    textcolor = 'w'
                else:
                    textcolor = 'k'
                if surf.T[jjx,iix] > 10:
                    annotation = r'$%.1f$'%surf.T[jjx,iix]
                else:
                    annotation = r'$%.2f$'%surf.T[jjx,iix]
                text = ax.text(ii, jj, annotation,
                               ha='center', va='center', color=textcolor)

    if savefig:
        if solver == 'valueiter' or solver == 'policyiter':
            filename = 'MANUSCRIPT_figs/advantage_'+solver
        else:
            raise ValueError('Unknown solver')
        filename += '_n%d_ps%.3f_tolerance%s_randomseed%s.pdf'%(n,
                                         p_s,tolerance,randomseed)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #

if __name__=='__main__':
    v0_evol, state_info, exe_time = find_optimal(3,1,2)

    for e in v0_evol:
        plt.plot(e)

    for e in agent.state_info:
        print(e['state'], e['action_space'], e['policy'])


































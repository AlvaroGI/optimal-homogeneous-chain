from copy import deepcopy
import functools
from itertools import combinations
import numpy as np
from pathlib import Path
import pickle

class Environment:
    '''Environment for our policy iteration algorithm.
        ---Inputs---
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) memory cutoff in time steps - entangled links
                        whose age reaches the cutoff are discarded.
        ---Inputs/Outputs to methods---
            · state:    (n-by-n array) matrix representation of the state.
                        Element ij contains the age of the entangled link
                        shared between nodes i and j. If the link does
                        not exist, the age is inf. In policy iteration,
                        we consider states just before swaps are performed.
            · action:   (list) set of nodes that must perform swaps.'''
    def __init__(self, p, cutoff, p_s=1):
        self.p = p
        self.cutoff = cutoff
        self.p_s = p_s

    def check_symmetry(self, state, rtol=0, atol=1e-10):
        '''Check if state is symmetric.'''
        assert np.allclose(state, state.T, rtol=rtol, atol=atol), 'state not symmetric'
        return

    def check_swap_action(self, state, action):
        '''Check if action can be performed on state.'''
        valid_action = True
        for mid_node in action:
            valid_action *= (len(np.where(state[mid_node]<np.infty)[0])==2)
        return valid_action

    def check_e2e_path(self, state):
        '''Return True if there is a path of virtual links from one end
            node to the other, and False otherwise.'''
        i = 0
        j = 0
        N = len(state)
        while i<N-1 and j<N-1:
            if state[i][j] == np.infty:
                j += 1
            else:
                i = j
        assert j==N-1 # Make sure we checked until the end of the chain
        if not state[i][j] == np.infty:
            e2e = True
        else:
            e2e = False
        return e2e

    def check_e2e_link(self, state):
        '''Return True if there is a virtual link between the end
            nodes, and False otherwise.'''
        if state[0][-1] == np.infty:
            e2e = False
        else:
            e2e = True
        return e2e

    def cutoffs(self, state):
        '''Remove links older than the cutoff.'''
        old_links = np.multiply((state>=self.cutoff),(state<np.infty)).nonzero()
        for i in range(len(old_links[0])):
            state[old_links[0][i],old_links[1][i]] = np.infty
            state[old_links[1][i],old_links[0][i]] = np.infty
        return state

    def find_available_links(self, state):
        '''Find every physical link with qubits available
            for elementary link generation in state.
            Return a list of nodes k that are ready to generate
            entanglement with node k+1.'''
        available_links = []
        k = len(state)-1 # Number of physical links
        for link in range(k):
            if (state[link,link+1] == np.infty # Link is free
            and (state[link,link+1:] == np.infty).all() # Right memory free
            and (state[:link,link+1] == np.infty).all()): # Left memory free
                available_links += [link]
        return available_links

    def find_virtual_neighbors(self, state, node):
        '''Return the list of nodes that share an entangled
            link with node in state.'''
        return np.where(state[node]<np.infty)[0]

    def generate_action_space(self, state):
        '''Generate list of valid actions from state.'''
        # Get valid actions consisting on one swap
        single_swaps = []
        for node in range(1, state.shape[0]-1):
            valid_action = self.check_swap_action(state, [node])
            if valid_action: single_swaps.append(node)
        # Add "no action"
        action_space = [[]]

        for n in range(1, len(single_swaps)+1):
            combis = [list(c) for c in combinations(single_swaps, n)]
            action_space += combis
        return action_space

    def generate_transitions(self, state):
        '''Return all states that could be obtained after attempting elementary
            link generation on every physical link with qubits available;
            the probability of obtaining each of those states;
            and the action space from each state'''
        s_out = []
        p_out = []
        a_out = []

        available_links = self.find_available_links(state)

        # Case in which all entanglement generation attempts fail
        s_out += [state]
        p_out += [(1-self.p)**(len(available_links))]
        a_out += [self.generate_action_space(state)]

        # Loop over all combination sizes of available links
        for number_GEN in range(1,len(available_links)+1):
            # Loop over all combinations of available links with fixed size
            combis_GEN = combinations(available_links, number_GEN)
            for combi_GEN in combis_GEN:
                _s = deepcopy(state)
                for link in combi_GEN:
                    _s[link,link+1] = 0
                    _s[link+1,link] = 0
                s_out += [_s]
                p_out += [self.p**len(combi_GEN) *
                          (1-self.p)**(len(available_links)-len(combi_GEN))]
                a_out += [self.generate_action_space(_s)]
        return s_out, p_out, a_out

    def generate_transitions_swaps(self, state, action):
        '''Return all states that could be obtained after succeeding/failing
            each swap; the probability of obtaining each of those states;
            and the action space from each state.'''
        s_out = []
        p_out = []
        a_out = []

        # Find all pairs of end nodes in a swap
        end_nodes_list = []
        for mid_node in action:
            end_nodes_list += [self.find_virtual_neighbors(state, mid_node)]

        # Case in which all swaps fail
        _s = deepcopy(state)
        for mn_idx, mid_node in enumerate(action):
            # If fail, remove input links
            _s[mid_node, end_nodes_list[mn_idx][0]] = np.infty
            _s[end_nodes_list[mn_idx][0], mid_node] = np.infty
            _s[mid_node, end_nodes_list[mn_idx][1]] = np.infty
            _s[end_nodes_list[mn_idx][1], mid_node] = np.infty
        s_out += [_s]
        p_out += [(1-self.p_s)**(len(action))]
        a_out += [self.generate_action_space(_s)]

        # Loop over all combinations of successful swaps
        for number_successes in range(1, len(action)+1):
            # Loop over all combinations of swaps with fixed number of successes
            combis_success = combinations(action, number_successes)
            for combi_success in combis_success:
                _s = deepcopy(state)
                for mid_node_idx, mid_node in enumerate(action):
                    end_nodes = self.find_virtual_neighbors(_s, mid_node)
                    if len(end_nodes)<2: # This can happen if a swap failed somewhere
                        if end_nodes[0]<mid_node:
                            end_nodes = [end_nodes[0], mid_node]
                        else:
                            end_nodes = [mid_node, end_nodes[0]]
                    if mid_node in combi_success:
                        # If success, apply swaps
                        _s = self.swap(_s, mid_node, end_nodes) # XXX: this should output s_out, p_out
                    else:
                        # If fail, remove input links
                        _s[mid_node, end_nodes[0]] = np.infty
                        _s[end_nodes[0], mid_node] = np.infty
                        _s[mid_node, end_nodes[1]] = np.infty
                        _s[end_nodes[1], mid_node] = np.infty
                s_out += [_s]
                p_out += [self.p_s**len(combi_success) *
                          (1-self.p_s)**(len(action)-len(combi_success))]
                a_out += [self.generate_action_space(_s)]

        assert np.abs(sum(p_out)-1) < 1e-10 # Normalization

        return s_out, p_out, a_out

    def step(self, current_state, action):
        '''Evolves the current state over a time slot. In our policy
            iteration algorithm, we apply the action (i.e., perform
            swaps) and then apply cutoffs, increase age of all links,
            and attempt entanglement generation.
            Return all states that could be obtained after the
            time slot, depending on what swaps succeeded/failed;
            the probability of obtaining each of those states;
            and the action space from each state.'''
        state = deepcopy(current_state)

        # Check symmetry of the input state
        self.check_symmetry(state)

        # Check if the action can be applied to the input state
        try:
            assert self.check_swap_action(state, action), 'A_swap not valid'
        except:
            print("Error: action cannot be applied to state")
            print(state, action)

        # Deterministic swaps
        if self.p_s == 1:
            # Apply swaps
            for mid_node in action:
                end_nodes = self.find_virtual_neighbors(state, mid_node)
                state = self.swap(state, mid_node, end_nodes)

            if not self.check_e2e_link(state):
                # Apply cutoffs
                state = self.cutoffs(state)

                # Advance time
                state += 1

                # Elementary link generation
                s_out, p_out, a_out = self.generate_transitions(state)
            else:
                s_out = [state]
                p_out = [1]
                a_out = [[]]

        # Probabilistic swaps
        else:
            # Apply swaps
            s_out0, p_out0, a_out0 = self.generate_transitions_swaps(state, action)

            assert len(s_out0)==len(p_out0)
            assert len(s_out0)==len(a_out0)

            # XXX: Remove duplicated states?
            s_out = []
            p_out_ = []
            a_out = []

            for idx0, state in enumerate(s_out0):
                if not self.check_e2e_link(state):
                    # Apply cutoffs
                    state = self.cutoffs(state)

                    # Advance time
                    state += 1

                    # Elementary link generation
                    s_out1, p_out1, a_out1 = self.generate_transitions(state)
                    s_out += s_out1
                    p_out_ += [pout1*p_out0[idx0] for pout1 in p_out1]
                    a_out += a_out1
                    # XXX: Remove duplicated states?
                else:
                    s_out += [state]
                    p_out_ += [p_out0[idx0]]
                    a_out += [[]]

            assert np.abs(sum(p_out_)-1) < 1e-10
            p_out = p_out_
        return s_out, p_out, a_out

    def swap(self, state, mid_node, end_nodes):
        '''Performs a deterministic swap on state, using mid_node (int) as
            the middle node and end_nodes (tuple) as end nodes.
            The age of the new link is the age of the oldest input link.'''
        if self.p_s == 1:
            assert end_nodes[0] < mid_node and mid_node < end_nodes[1], 'mid_node \
                  should not share entangled links with its left/right side only'

        # Create new link
        state[end_nodes[0],end_nodes[1]] = max(state[end_nodes[0],mid_node],
                                         state[mid_node, end_nodes[1]])
        state[end_nodes[1],end_nodes[0]] = state[end_nodes[0],end_nodes[1]]

        # Remove input links
        state[mid_node, end_nodes[0]] = np.infty
        state[mid_node, end_nodes[1]] = np.infty
        state[end_nodes[0], mid_node] = np.infty
        state[end_nodes[1], mid_node] = np.infty

        return state

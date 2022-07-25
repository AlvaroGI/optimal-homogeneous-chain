import numpy as np

class Agent:
    '''Agent for our policy iteration algorithm. This agent stores
        all the information about the current policy.
        ---Arguments---
            · s0:   (n-by-n array) initial state in matrix form.
            · state_list: (list of vectors) list of states visited
                          by the algorithm, in vector form.
            · state_info: (list of dicts) each dictionary corresponds to
                          a different state visited by the algorithm,
                          and it contains the following keys:
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
                         calculated as -(value+1).'''
    def __init__(self, s0, max):
        self.max = max
        self.init_value = 0
        # Transform initial state s0 into compact form and save
        vec_state = self.state2vec(s0)
        self.state_list = [vec_state]
        self.state_info = [ {"state": s0,
                             "policy": [1.],
                             "action_space": [[]],
                             "value": self.init_value} ]

    def get(self, idx, key):
        return  self.state_info[idx][key]

    def get_state_idx(self, state):
        '''Get index of state in state_list and state_info.'''
        idx = None
        state_is_new = False

        # First transform state into a compact form
        vec_state = self.state2vec(state)

        # Look for state in state list
        for j, s in enumerate(self.state_list):
            if np.array_equal(s, vec_state):
                idx = j
                break

        if idx is None:
            state_is_new = True

        return idx, state_is_new

    def init_policy(self, action_space):
        '''Initialize action performed on a state with action_space
            elements as the only valid actions. The initialization
            is a random choice among all actions.'''
        if len(action_space) < 2:
            policy = [1.]
        else:
            num_valid_actions = len(action_space)
            policy = [1/num_valid_actions for j in range(num_valid_actions)]
        return policy

    def observe(self, state, action_space=None):
        '''If state is new, add to state_list and state_info.'''
        idx, state_is_new = self.get_state_idx(state)

        if state_is_new:
            assert action_space is not None
            idx = len(self.state_list)
            vec_state = self.state2vec(state)
            policy = self.init_policy(action_space)
            value = self.init_value if state[0,-1] == np.infty else 0.0
            self.state_list.append(vec_state)
            self.state_info.append({"state": state,
                                    "policy": policy,
                                    "action_space": action_space,
                                    "value": value})

        return idx

    def state2vec(self, state):
        '''Map state from matrix representation to vector representation.'''
        vec_state = np.hstack([state[j,j+1:] for j in range(state.shape[0])])
        for j, e in enumerate(vec_state):
            if e == np.infty: vec_state[j] = self.max+1
        vec_state = vec_state.astype(np.int32)
        return vec_state

    def update(self, idx, update, key):
        '''Update value of state with index idx.'''
        self.state_info[idx][key] = update

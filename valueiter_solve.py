#!/usr/bin/python
from math import *
import numpy as np
import itertools
import sympy as sym
import pickle
import sys
import pdb
import argparse
import os
import main

#------------------------------------------------------------------------------
# ARGPARSER
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default=3,
                    help='Number of nodes.')
parser.add_argument('--p', type=float, default=1,
                    help='Probability of successful elementary link generation.')
parser.add_argument('--p_s', type=float, default=1,
                    help='Probability of successful swap.')
parser.add_argument('--cutoff', type=int, default=2,
                    help='Cutoff time.')
parser.add_argument('--tol', type=float, default=1e-7,
                    help='Tolerance.')
parser.add_argument('--randomseed', type=int, default=2,
                    help='Random seed.')
parser.add_argument('--print_output', type=int, default=0,
                    help='If 1, prints output.')

FLAGS = parser.parse_args()

if FLAGS.n < 3:
    raise ValueError('Number of nodes too small')

#------------------------------------------------------------------------------
# PARAMETERS
#------------------------------------------------------------------------------
n = FLAGS.n
# Number of links
n_links = n - 1
# Cutoff
cutoff = FLAGS.cutoff

pVal = FLAGS.p
swpVal = FLAGS.p_s

tolerance = FLAGS.tol
randomseed = FLAGS.randomseed

print_output = FLAGS.print_output

# Create data directory if needed
try:
    os.mkdir('data_valueiter')
except FileExistsError:
    pass
try:
    os.mkdir('data_valueiter/solution')
except FileExistsError:
    pass

#------------------------------------------------------------------------------
# CALCULATIONS
#------------------------------------------------------------------------------
if main.check_valueiter_data(n, pVal, swpVal, cutoff, tolerance, randomseed):
    print('Solution already exists!')
else:
    np.random.seed(randomseed)
    # Solve system of equations (value iteration)
    if True:
        # specify link-level entanglement generation success prob.
        p = sym.Symbol('p')
        # specify entanglement swapping success probability
        swp = sym.Symbol('swp')
        # get the states and equations
        model_filenames = 'data_valueiter/model/MDPmodel_n%s_tc%s'%(n,cutoff)
        S = np.load(model_filenames+'_states.npy')
        linkLabels = np.load(model_filenames+'_linkLabels.npy')
        Eqns = pickle.load(open(model_filenames+'_equations.pkl','rb'))

        Ts = {}
        # get rid of unnecessary states (those that have at least one link with age >= cutoff)
        r = 0
        sidx = 0
        while r < S.shape[0]:
            Srow = S[r]
            if cutoff in Srow:
                S = np.delete(S,r,0)
            else:
                Ts[sidx] = np.random.uniform(0,1,1)[0]
                r += 1
                
            sidx += 1

        numStates = S.shape[0]
        # initialize Ts to random values
        #Ts = np.random.uniform(0,1,numStates)
        # initialize the equations by substituting p
        for T in Ts:
            relKeys = [(i,j) for (i,j) in Eqns if i == T]
            for key in relKeys:
                partialEqn = Eqns[key]
                for oidx in range(Eqns[key].shape[0]):
                    for iidx in range(Eqns[key][:,3][oidx].shape[0]):
                        Eqns[key][:,3][oidx][iidx] = Eqns[key][:,3][oidx][iidx].subs([(swp, swpVal),(p, pVal)])
        # start the optimization procedure
        delta = tolerance+1
        while delta > tolerance:
            delta = 0
            
            for T in Ts:
                T_so_far = 0
                relKeys = [(i,j) for (i,j) in Eqns if i == T]
                for key in relKeys:
                    partialEqn = Eqns[key]
                    expDelTimes = np.empty((0,partialEqn.shape[0]))
                    
                    for row in partialEqn:
                        # update with the latest values of T
                        partialTs = [Ts.get(rowkey) for rowkey in row[0].astype(int)]
                        #partialTs = np.concatenate(partialTs, axis=0)
                        # compute the expected delivery time for this decision
                        expDelT = np.inner(partialTs,row[3])
                        expDelTimes = np.append(expDelTimes,expDelT)
                    
                    # get the index of the minimum T value
                    minValIdx = np.argmin(expDelTimes,0)
                    # update decision variables accordingly
                    newDecVars = [0]*(partialEqn.shape[0])
                    newDecVars[minValIdx] = 1
                    
                    Eqns[key][:,2] = newDecVars
                    # update the value of T
                    T_so_far += expDelTimes[minValIdx]
                
                T_so_far += 1
                delta = max(delta,abs(Ts[T]-T_so_far))
                Ts[T] = T_so_far

    equations = []
    for T in range(len(Ts)):
        relKeys = [(i,j) for (i,j) in Eqns if i == T]
        eqs_T = []
        if print_output == 1:
            print('T is '+str(T))
        for key in relKeys:
            eqs_T += [Eqns[key]]
            if print_output == 1:
                print(Eqns[key])
        equations += [eqs_T]

    # Save data
    data = {'state_labels': linkLabels,
            'states': S,
            'policy': equations,
            'values': Ts}
    solution_filename = ('data_valueiter/solution/MDPsol_n%s_p%.3f'%(n,pVal)+
                         '_ps%.3f_tc%s_tol%s_randomseed%s'%(swpVal,cutoff,tolerance,randomseed))

    with open(solution_filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

















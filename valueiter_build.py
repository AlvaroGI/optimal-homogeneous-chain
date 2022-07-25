#!/usr/bin/python
from math import *
import numpy as np
import itertools
from itertools import islice
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
parser.add_argument('--cutoff', type=int, default=2,
                    help='Cutoff time.')
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

print_output = FLAGS.print_output

# Create data directory if needed
try:
    os.mkdir('data_valueiter')
except FileExistsError:
    pass
try:
    os.mkdir('data_valueiter/model')
except FileExistsError:
    pass

#------------------------------------------------------------------------------
# CALCULATIONS
#------------------------------------------------------------------------------
if main.check_valueiter_model(n, cutoff):
    print('Model already exists!')
else:
    # Generate equations
    if True:
        # declare p and define q
        p = sym.Symbol('p')
        q = 1-p
        # declare a variable for probabilistic entanglement swapping
        swp = sym.Symbol('swp')

        # determine the vector size for the state representation
        k = ((n_links+1)**2-(n_links+1)-2)//2

        # link label list
        linkLabels = []
        # special links in the chain (links which preserve state symmetry)
        symLinkLabels = []
        breakFlag = 0
        appIdx = 0

        # create link labels (i,j)
        for i1 in range(1,n_links+1):
            if breakFlag:
                break
            for i2 in range(i1+1,n_links+2):
                if (i1,i2) in linkLabels:
                    break
                else:
                    correspLink = (n_links-i2+2,n_links-i1+2)
                    if (i1,i2) == correspLink:
                        symLinkLabels.append((i1,i2))
                    else:
                        linkLabels.insert(appIdx,(i1,i2))
                        linkLabels.insert(len(linkLabels)-appIdx,correspLink)
                        appIdx += 1

        # Remove (1,n_links+1)
        symLinkLabels.remove((1,n_links+1))
        numSymLinks = len(symLinkLabels)
        # Add special links to the end
        linkLabels.extend(symLinkLabels)

        # indices of elementary links within the linkLabels list
        elemLinkIdxs = []
        idx = 0
        for tpl in linkLabels:
            if tpl[1]-tpl[0] == 1:
                    elemLinkIdxs.append(idx)
            idx = idx+1

        # state matrix
        S = np.array([np.ones(k)*-1])
        # keeping track of T_i's
        TVars = []
        # a data structure to keep track of the equations for each T
        Eqns = {}
        # decision variables
        DecVars = {}

        # keep track of state types (symmetric = 1, non-symmetric = 0)
        stateTypes = {}
        stateTypes[0] = 1

        # check whether a state already exists; output the state index (or create a new one)
        def getStateIdxSVar(state):
            global S

            # check whether the state is symmetric or non-symmetric type,
            # and also whether it is new
            stateType, dupState = getStateInfo(state)
            isNewState = 0
            rowMatch = np.where(np.all(S==state,axis=1))[0]
            if stateType == 1 and rowMatch.size == 0: # new state, symmetric
                isNewState = 1
            elif stateType == 0:
                # check the flipped version of this non-symmetric state too
                rowMatchDup = np.where(np.all(S==dupState,axis=1))[0]
                if rowMatch.size == 0 and rowMatchDup.size == 0:
                    # this non-symmetric state is new
                    isNewState = 1
            
            if isNewState:
                # add it to the table
                S = np.vstack([S,state])
                # create an index for this state
                stateIdx = S.shape[0]-1
                # create a sym var, if necessary
                if cutoff not in state:
                    sVar = sym.Symbol("T"+str(stateIdx))
                else:
                    sVar = -1
                TVars.append(sVar)
                stateTypes[stateIdx] = stateType
            else:
                if rowMatch.size == 1:
                    stateIdx = rowMatch[0]
                else:
                    stateIdx = rowMatchDup[0]
                sVar = TVars[stateIdx]

            return stateIdx, sVar
            
        # Get state type: 1 for symmetric, 0 for non-symmetric;
        # also get the duplicate version of this state.
        def getStateInfo(state):
            # flip the state but ignore links that don't affect the state's symmetry
            dupState = np.copy(state)
            
            dupState[:k-numSymLinks] = dupState[k-numSymLinks-1::-1]
            comp = state[:k-numSymLinks] == dupState[:k-numSymLinks]
            stateType = comp.all()
            
            return stateType, dupState

        # get the swap info of a state, if swap is done at a given node
        # output format: leftLinkIdx, rightLinkIdx, newLinkIdx, newLinkAge
        def getSwapInfo(node,posLinks,state):
            # get the left and right links
            leftLink = {(a,b) for (a,b) in posLinks if b == node}.pop()
            rightLink = {(a,b) for (a,b) in posLinks if a == node}.pop()
            # get the indices of the left and right links
            leftLinkIdx = linkLabels.index(leftLink)
            rightLinkIdx = linkLabels.index(rightLink)
            
            linkAge = max(state[leftLinkIdx],state[rightLinkIdx])
            
            return leftLinkIdx, rightLinkIdx, linkAge

        # determine swap groups: groups of nodes that are involved in swapping
        # operations, such that if one node fails a swap, then the group fails as a whole
        def getSwapGroups(swapNodes,posLinks,state):
            swapGroups = {}
            # keep track of the ages of the new links
            newLinkAges = []
            for sNode in swapNodes:
                # since this is a swapNode, it will have
                # a left neighbor and a right neighbor
                leftLinkIdx, rightLinkIdx, linkAge = getSwapInfo(sNode,posLinks,state)
                
                leftNeighbor = linkLabels[leftLinkIdx][0]
                rightNeighbor = linkLabels[rightLinkIdx][1]
                sNodeGroupIdx = 0
                for gIdx in swapGroups:
                    if leftNeighbor in swapGroups[gIdx] or rightNeighbor in swapGroups[gIdx]:
                        swapGroups[gIdx].append(sNode)
                        sNodeGroupIdx = gIdx
                        newLinkAges[gIdx-1] = max(newLinkAges[gIdx-1],linkAge)
                        break
                if sNodeGroupIdx == 0:
                    sNodeGroupIdx = len(swapGroups)+1
                    swapGroups[sNodeGroupIdx] = [sNode]
                    newLinkAges.append(linkAge)
            
            return swapGroups, newLinkAges

        # determine if a state can potentially produce an e2e link;
        # if so, return the nodes that should be swapped to achieve this
        def getStateE2Einfo(state):
            ise2e = 0
            rnode = [linkLabels[i][1] for i in range(k) if linkLabels[i][0] == 1 and state[i] > -1]
            swapNodes_e2e = []
            
            while rnode:
                if rnode[0] == n_links+1:
                    ise2e = 1
                    break
                else:
                    swapNodes_e2e.append(rnode[0])
                    lnode = rnode[0]
                    rnode = [linkLabels[i][1] for i in range(k) if linkLabels[i][0] == lnode and state[i] > -1]
            
            if ise2e:
                return swapNodes_e2e
            else:
                return []

        # get info about the state we are attempting to transition to,
        # i.e., where all swaps are successful
        def getAllSucTransStateIdxSVar(nodes,origState,posLinks):
            state = np.array(origState)
            # perform successful swaps
            for node in nodes:
                leftLinkIdx, rightLinkIdx, linkAge = getSwapInfo(node,posLinks,state)
                # check whether the state is an e2e state
                newLink = (linkLabels[leftLinkIdx][0], linkLabels[rightLinkIdx][1])
                if newLink == (1,n_links+1):
                    return -1, -1
                else:
                    state[leftLinkIdx] = -1
                    state[rightLinkIdx] = -1
                    newLinkIdx = linkLabels.index(newLink)
                    state[newLinkIdx] = linkAge
            
            # get the state index and variable for the final state
            return getStateIdxSVar(state)

        # get indices of absent elementary links that can get generated,
        # i.e., the communication qubits are not occupied
        def getAbsentLinkIdxs(state):
            absentElemLinkIdxs = [idx for idx in elemLinkIdxs if state[idx] == -1]
            absentElemLinkIdxsRegen = []
            for idx in absentElemLinkIdxs:
                elemLink = linkLabels[idx]
                relIdxsLeft = [x for x in range(0,k) if elemLink[0] == linkLabels[x][0]]
                relIdxsRight = [x for x in range(0,k) if elemLink[1] == linkLabels[x][1]]
                nonElemPresentLinkIdxs = [x for x in relIdxsLeft+relIdxsRight if state[x] > -1]
                if len(nonElemPresentLinkIdxs) == 0:
                    absentElemLinkIdxsRegen.append(idx)
            
            return absentElemLinkIdxsRegen

        # initialize the state index
        stateIdx = 0
        while S.shape[0] >= stateIdx+1:
            sCur = S[stateIdx]
            #if stateIdx == 2:
            #    pdb.set_trace()
            # if any of the links are at the cutoff age, don't process this state
            if cutoff in sCur: #or len(getStateE2Einfo(sCur)) == 1:
                stateIdx += 1
                continue
            # make a new variable for this state, if there isn't one
            if stateIdx+1 > len(TVars):
                sVarCur = sym.Symbol("T"+str(stateIdx))
                TVars.append(sVarCur)
            else:
                sVarCur = TVars[stateIdx]
            
            # for the links that are present, we will add 1 to their ages
            presentLinkIdxs = [idx for idx in range(0,k) if sCur[idx] > -1]
            # for the elementary links that are absent, any combination
            # of them can be generated in the next time step, as long as the memories
            # are not currently occupied; get the indices of these absent elementary links:
            absentLinkIdxs = getAbsentLinkIdxs(sCur)
            
            # consider all combinations of link presence/absence:
            numAbsentLinks = len(absentLinkIdxs)
            allCombos = numAbsentLinks*[[0,1]]
            combo_iterator = itertools.product(*allCombos)
            
            # start iterating through the possible state transitions,
            # ignoring those that are duplicates which have already been encountered
            stateIdxsSeen = []
            comboIdx = 0
            for combo in combo_iterator:
                newState = np.array(sCur)
                np.add.at(newState,absentLinkIdxs,combo)
                np.add.at(newState,presentLinkIdxs,1)
                
                # check whether this state has already been processed
                newStateIdx, newStateSVar = getStateIdxSVar(newState)
                
                if newStateIdx in stateIdxsSeen:
                    # this state has already been processed
                    continue
                else:
                    stateIdxsSeen.append(newStateIdx)
                
                # compute the probability of arriving to this state
                numSucc = sum(combo)
                transProb = (p**numSucc)*(q**(len(combo)-numSucc))
                
                # if this state can potentially produce an e2e entanglement,
                # determine which nodes are necessary to swap
                criticalNodes = getStateE2Einfo(newState)
                numCriticalNodes = len(criticalNodes)
                if numCriticalNodes == 1:
                    necessarySwaps = criticalNodes
                else:
                    necessarySwaps = []
                
                # check if there are any decisions to make
                possibleSwaps = [] # nodes that can do swaps
                possibleSwapLinks = set() # links associated with possible swaps
                for nodeNum in range(2,n_links+1):
                    leftPresentLink = [lidx for lidx in range(0,k) if newState[lidx] > -1 and linkLabels[lidx][1] == nodeNum]
                    rightPresentLink = [lidx for lidx in range(0,k) if newState[lidx] > -1 and linkLabels[lidx][0] == nodeNum]
                    if len(leftPresentLink) and len(rightPresentLink):
                        if nodeNum not in necessarySwaps:
                            possibleSwaps.append(linkLabels[leftPresentLink[0]][1])
                        possibleSwapLinks.add(linkLabels[leftPresentLink[0]])
                        possibleSwapLinks.add(linkLabels[rightPresentLink[0]])
                
                # then, consider all combinations of swap/wait
                numSwapNodes = len(possibleSwaps)
                swapCombos = numSwapNodes*[[0,1]]
                swapCombo_iterator = itertools.product(*swapCombos)
                # keep track of possible transition states (idx = first, sym var = second column),
                # corresponding decision variables (third column),
                # and the multiplier x transProb for each state (depending on the transition type, fourth column)
                transStates_decVars = np.empty((0,4))
                # determine the appropriate multiplier value
                mult = 1
                if stateTypes[stateIdx] == 1 and stateTypes[newStateIdx] == 0:
                    # if transition is from a symmetric state and through a non-symmetric
                    # state, then need a multiplier of two
                    mult = 2
                #if stateIdx == 17:
                    #pdb.set_trace()
                for swapCombo in swapCombo_iterator:
                    swapNodes = [possibleSwaps[swap_idx] for swap_idx in range(0,numSwapNodes) if swapCombo[swap_idx] == 1]
                    # if this is an e2e state, add the necessary swap nodes as well
                    swapNodes = swapNodes + necessarySwaps
                    # TODO: figure out state trying to transition to, disallow if it's 0 and an e2e swap is available
                    # index of the state that we are attempting to transition to
                    allSucTransStateIdx, allSucTransStateSVar = getAllSucTransStateIdxSVar(swapNodes,newState,possibleSwapLinks)
                    if numCriticalNodes > 0 and allSucTransStateIdx == 0:
                        continue
                    swapGroups, newLinkAges = getSwapGroups(swapNodes,possibleSwapLinks,newState)
                    swapGroupCombos = len(swapGroups)*[[0,1]]
                    swapGroup_iterator = itertools.product(*swapGroupCombos)
                    
                    # assign a temporary value to the decision variable field
                    newRow = np.array([np.array([]),np.array([]),1,np.array([])],dtype=object)
                    # some swap groups will succeed, others will fail
                    for sgCombo in swapGroup_iterator:
                        ise2eSwap = 0
                        
                        transState = np.array(newState)
                        
                        eventProb = 1
                        gIdx = 1
                        # "perform" the swaps and compute the probability of this event
                        for g in sgCombo:
                            groupNodes = swapGroups[gIdx]
                            if g:
                                eventProb *= swp**len(groupNodes)
                                # perform a successful swap
                                newLink = [inf,0]
                                for lIdx in range(k):
                                    if transState[lIdx] > -1 and linkLabels[lIdx][0] in groupNodes:
                                        newLink[1] = max(newLink[1],linkLabels[lIdx][1])
                                        transState[lIdx] = -1
                                    elif transState[lIdx] > -1 and linkLabels[lIdx][1] in groupNodes:
                                        newLink[0] = min(newLink[0],linkLabels[lIdx][0])
                                        transState[lIdx] = -1
                                newLink = (newLink[0],newLink[1])

                                if newLink == (1,n_links+1):
                                    # this is an end-to-end swap
                                    ise2eSwap = 1
                                    break
                                else:
                                    newLinkIdx = linkLabels.index(newLink)
                                    transState[newLinkIdx] = newLinkAges[gIdx-1]
                            else:
                                eventProb *= 1-swp**len(groupNodes)
                                # perform an unsuccessful swap
                                for lIdx in range(k):
                                    if linkLabels[lIdx][0] in groupNodes or linkLabels[lIdx][1] in groupNodes:
                                        transState[lIdx] = -1
                            
                            gIdx += 1
                         
                        if not ise2eSwap:
                            # remove links that are at the cutoff
                            transState[transState == cutoff] = -1
                        
                            # get the index and symbolic variable of this state
                            transStateIdx, transStateSVar = getStateIdxSVar(transState)
                            # if this is the state we are attempting to transition to,
                            # get its index and decision variable
                            #if sum(sgCombo) == len(sgCombo):
                            #    allSucTransStateIdx, allSucTransStateSVar = transStateIdx, transStateSVar

                            newRow[0] = np.append(newRow[0],transStateIdx)
                            newRow[1] = np.append(newRow[1],transStateSVar)
                            newRow[3] = np.append(newRow[3],mult*transProb*eventProb)
                        
                    if numSwapNodes == 0:
                        # don't need a decision variable in this case
                        transStates_decVars = np.vstack((transStates_decVars, newRow))
                    elif (newStateIdx,allSucTransStateIdx) not in DecVars:
                        #pdb.set_trace()
                        DecVars[newStateIdx,allSucTransStateIdx] = newStateIdx
                        # create a new symbolic variable for this transition
                        decVar = sym.Symbol("s_"+str(newStateIdx)+"_"+str(allSucTransStateIdx))
                        newRow[2] = decVar
                        transStates_decVars = np.vstack((transStates_decVars, newRow))
                
                if len(transStates_decVars) > 0:
                    if transStates_decVars.shape[0]  == 1:
                        transStates_decVars[0,2] = 1
                    Eqns[stateIdx,comboIdx] = transStates_decVars
                    
                comboIdx += 1
            stateIdx += 1

    # Save model
    filenames = 'data_valueiter/model/MDPmodel_n%s_tc%s'%(n,cutoff)
    np.save(filenames+'_states.npy',S)
    np.save(filenames+'_linkLabels.npy',linkLabels)
    f = open(filenames+'_equations.pkl','wb')
    pickle.dump(Eqns,f)
    f.close()

    # print the model
    if print_output == 1:
        for T in range(len(TVars)):
            relKeys = [(i,j) for (i,j) in Eqns if i == T]
            print("T is "+str(T))
            for key in relKeys:
                print(Eqns[key])

        print("The link labels are:")
        print(linkLabels)
        print("The states are:")
        print(S)

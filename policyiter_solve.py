import numpy as np
from policy import Agent
from environment import Environment
from tqdm import tqdm
import main
import argparse

#------------------------------------------------------------------------------
# ARGPARSER
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default='3',
                    help='Number of nodes.')
parser.add_argument('--p', type=float, default='1',
                    help='Probability of successful elementary link generation.')
parser.add_argument('--p_s', type=float, default='1',
                    help='Probability of successful swap.')
parser.add_argument('--cutoff', type=int, default='2',
                    help='Cutoff time.')
parser.add_argument('--tol', type=float, default='1e-5',
                    help='Tolerance.')
parser.add_argument('--progress', type=bool, default='True',
                    help='If True, shows progress.')
parser.add_argument('--policy', type=str, default='optimal',
                    help='"optimal" or "swap-asap".')

FLAGS = parser.parse_args()


#------------------------------------------------------------------------------
# PARAMETERS
#------------------------------------------------------------------------------
n = FLAGS.n
p = FLAGS.p
p_s = FLAGS.p_s
cutoff = FLAGS.cutoff
tolerance = FLAGS.tol
progress = FLAGS.progress

if n < 3:
    raise ValueError('Number of nodes too small')

policy = FLAGS.policy

#------------------------------------------------------------------------------
# CALCULATIONS
#------------------------------------------------------------------------------
if policy == 'optimal':
    # Check if data exists
    if not main.check_policyiter_data(n, p, p_s, cutoff, tolerance):
        # Find optimal protocol
        v0_evol, state_info, exe_time = main.policy_iteration(n, p, p_s, cutoff,
                                                              tolerance=tolerance,
                                                              progress=progress,
                                                              savedata=True)
        # Save data
        main.save_policyiter_data(n, p, p_s, cutoff, tolerance, v0_evol, state_info, exe_time)
        print('Done! Data saved!')
    else:
        print('Data already exists!')
elif policy == 'swap-asap':
    # Check if data exists
    if not main.check_swapasapDP_data_exists(n, p, p_s, cutoff, tolerance):
        # Find optimal protocol
        v0_evol, state_info, exe_time = main.find_swapasap(n, p, cutoff, p_s=p_s,
                                                          tolerance=tolerance,
                                                          progress=progress,
                                                          savedata=True)
        # Save data
        main.save_swapasapDP_data(n, p, p_s, cutoff, tolerance, v0_evol, state_info, exe_time)
        print('Done! Data saved!')
    else:
        print('Data already exists!')
else:
    raise ValueError('Unknown policy')























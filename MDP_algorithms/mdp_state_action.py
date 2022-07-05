# -*- coding: utf-8 -*-

import numpy as np
import util as ut
import random


def pick_up_delivery_dynamics(Rows, Columns, rate, drop_offs, pick_ups, p=0.98):
    P_0 = ut.nonErgodicMDP(Rows, Columns, p=p, with_stay=True)
    S_half, S_halfA = P_0.shape
    A = int(S_halfA/S_half)
    S = S_half*2
    P = np.zeros((S, S*A))
    P[:S_half, :S_halfA] = 1*P_0
    P[S_half:, S_halfA:] = 1*P_0
    # transition from pickup to delivery
    # pick_up_row = range(S_half - Columns, S_half)
    for s in pick_ups:
        # sa = slice(s*A,(s+1)*A)
        orig_probability = P[s, :]*1
        P[s, :] = (1 - rate) * orig_probability
        P[s + S_half, :] += rate * orig_probability
        # P[s, s*A :(s+1)*A] = orig_probability[s*A:(s+1)*A] # can't enter drop off mode from drop off mode
    # delivery_row = range(S_half,  S_half+Columns)
    for s in drop_offs:
        P[s, :] += P[s + S_half, :]*1
        P[s + S_half, :] = 0
        
    return P

def pick_up_multi_delivery_dynamics(Rows, Columns, rate, delivery_dist, 
                                    pick_up, drop_off_states):
    P_0 = ut.nonErgodicMDP(Rows, Columns, p=0.98, with_stay=True)
    modes = len(delivery_dist)
    S_sec, _ = P_0.shape
    P = np.kron(np.eye(modes+1), P_0)
    # print(f'transition_sizes are {P.shape}')
    # print(f'transition current summation {np.ones(S_sec*(modes+1)).T.dot(P)}')
    # transition from pickup to delivery    
    drop_off_p = (1 - rate) * P[pick_up, :]
    P[pick_up, :] = rate * P[pick_up, :]
    # probability of entering delivery mode
    for m_ind in range(modes):
        P[pick_up + S_sec*(m_ind+1), :] += delivery_dist[m_ind] * drop_off_p

    # transition back to pick up from each delivery spot
    for mode in range(modes):
        delivery_state = drop_off_states[mode] + (mode+1)*S_sec
        P[drop_off_states[mode], :] += P[delivery_state, :]
        P[delivery_state, :] = 0
        
    return P

def transition_mat_to_tensor(P):
    S, SA = P.shape
    A = int(SA/S)
    P_tensor  = np.zeros((S,S,A))
    for s in range(S):
        for a in range(A):
            mat_ind = a + s*A
            P_tensor[:, s, a] = P[:, mat_ind]
    return P_tensor

def pick_up_multi_delivery_tensor(Rows, Columns, rate, delivery_dist, 
                                    pick_up, drop_off_states):
    P_0 = ut.nonErgodicMDP(Rows, Columns, p=0.98, with_stay=True)
    modes = len(delivery_dist)
    S_sec, _ = P_0.shape
    P = np.kron(np.eye(modes+1), P_0)
    # print(f'transition_sizes are {P.shape}')
    # print(f'transition current summation {np.ones(S_sec*(modes+1)).T.dot(P)}')
    # transition from pickup to delivery    
    drop_off_p = (1 - rate) * P[pick_up, :]
    P[pick_up, :] = rate * P[pick_up, :]
    # probability of entering delivery mode
    for m_ind in range(modes):
        P[pick_up + S_sec*(m_ind+1), :] += delivery_dist[m_ind] * drop_off_p

    # transition back to pick up from each delivery spot
    for mode in range(modes):
        delivery_state = drop_off_states[mode] + (mode+1)*S_sec
        P[drop_off_states[mode], :] += P[delivery_state, :]
        P[delivery_state, :] = 0
        
    return P

def probability_to_dict(S, A, P_arr):
    # probability dictionary has form:
    # key = (s, a), value = (probability, state_num)
    P_dict = {}
    for s in range(S):
        for a in range(A):
            P_dict[(s,a)] = []
            for s_dest in range(S):
                if P_arr[s_dest, s*A + a] > 0:
                    P_dict[(s,a)].append((P_arr[s_dest, s*A + a], s_dest))
    return P_dict

def pick_up_delivery_multi_cost(Rows, Columns, A, T, pick_up_state, deliveries, 
                                p_num, minimize=True):
    targ_rew = 0 if minimize else 1.
    S_sec = Rows*Columns
    S = S_sec * (len(deliveries) + 1)
    if minimize:
        C = np.ones((S, A, T+1))
    else:
        C = np.zeros((S, A, T+1))
    # cost for agents in pick up mode
    C[pick_up_state, :, :] = targ_rew
    # cost for agents delivery mode
    for mode in range(len(deliveries)):
        C[(mode+1)*S_sec + deliveries[mode], :, :] = targ_rew
    return C


def pick_up_delivery_cost(Rows, Columns, A, T, pick_up_state, drop_offs, p_num, 
                         minimize=True):
    targ_rew = 0 if minimize else 1.
    S_sec = Rows*Columns
    S = S_sec * 2
    if minimize:
        C = np.ones((S, A, T+1))
    else:
        C = np.zeros((S, A, T+1))
    # cost for agents in pick up mode
    C[pick_up_state, :, :] = targ_rew
        
    # cost for agents delivery mode
    for drop_off in drop_offs:
        C[drop_off + S_sec, :, :] = targ_rew    
    return C

def set_up_cost(Rows, Columns, A, T, target_col, target_row,  p_num, 
                minimize=True, scal=1.):
    targ_rew = 0 if minimize else scal
    S = Rows * Columns
    if minimize:
        C = [np.ones((S, A, T+1)) for _ in range(p_num)]
    else:
        C = [np.zeros((S, A, T+1)) for _ in range(p_num)]
    for p in range(p_num):
        C[p][target_row*Columns + target_col[p], :, :] = targ_rew
    return C
    
def pol2dist(policy, x, P, T): 
    # policy is a 3D array for player
    # x is player p's initial state distribution at t = 0
    # returns player P's final distribution
    S, SA, _ = policy.shape
    x_arr = np.zeros((len(x), T+1))
    x_arr[:, 0] = x
        
    markov_chains = np.einsum('ij, kjl->ikl', P, policy)
    # print(f'x_shape {x_arr.shape}')
    # print(f' markov chain shape {markov_chains[:, :, 0].shape}')
    for t in range(T):
        # x is the time state_density
        # print(f'x shape is {markov_chains[:, :, t].dot(x_arr[:, t]).shape}')
        x_arr[:, t+1] = markov_chains[:, :, t].dot(x_arr[:, t]) 
    # print(f'policy {policy.shape} x_arr {x_arr.shape}')
    # print(f' t is {T}')
    y = np.einsum('sat, st->at', policy, x_arr)

    return y

def state_congestion_faster(Rows, Columns, modes, A, T, y):
    scal = 40.
    c_cost = np.zeros((modes*Rows*Columns, A, T+1))
    S = modes*Rows*Columns
    sum_actions = np.kron(np.eye(S), np.ones(A).T)
    sum_modes = np.kron(np.ones(modes).T, np.eye(Rows*Columns))
    expand_modes = np.kron(np.eye(Rows*Columns), np.ones(modes)).T
    # print(f'sum_modes shape {sum_modes.shape}')
    physical_dist = sum_modes.dot(sum_actions).dot(y)
    # print(f'physical dist shape is {physical_dist.shape}')
    congestion = scal*np.exp(scal*(physical_dist - 1))
    expanded_congestion = expand_modes.dot(congestion)
    # print(f'congestion_shape {congestion.shape}')
    for a in range(A):
        c_cost[:, a, :]  = expanded_congestion
    return c_cost

def state_congestion(Rows, Columns, modes, A, T, y):
    c_cost = np.zeros((modes*Rows*Columns, A, T+1))
    
    for x_ind in range(Columns):
        for y_ind in range(Rows):
            common_states = [y_ind * Columns + x_ind]
            for mode in range(modes-1):
                common_states.append(common_states[-1] + Rows*Columns)
            for t in range(T+1):
                density = sum([y[s*A:(s+1)*A, t] for s in common_states])
                congestion = 5* np.exp(5 * (density - 1))
                for s in common_states:
                    c_cost[s, :, t] += congestion
    return c_cost

def start_at_locs(pols, p_num, drop_offs, P, S, T):
    initial_x = [np.zeros(S) for _ in range(p_num)]
    x = [[] for _ in range(p_num)]
    for p in range(p_num):
        initial_x[p][drop_offs[p]] = 1.
        x[p].append(pol2dist(pols[:,:,:,p], initial_x[p], P[p], T))
    return x, initial_x

def policy_list(policies, P, T, p_num, Columns):
    # policy = ut.random_initial_policy_finite(Rows, Columns, A, T+1, p_num)
    S, SA = P[0].shape
    initial_x = [np.zeros(S) for _ in range(p_num)]
    x_init_state  = random.sample(range(Columns), p_num)
    x = [[] for _ in range(p_num)]
    for p in range(p_num):
        initial_x[p][x_init_state[p]] = 1.
        x[p].append(pol2dist(policies[:,:,:,p], initial_x[p], P[p], T))
    return x, initial_x


def execute_policy(initial_x, P, pols, T, targets):
    S, SA = P[0].shape 
    S_half = int(S/2)
    A = int(SA/S)
    # ind of first position the players are in
    trajs = [[np.where(x == 1)[0][0]] for x in initial_x] 
    # print(f' traj is {trajs}')
    
    _, _, T, p_num = pols.shape
    flat_pols = np.sum(pols, axis=0)
    collisions = {p: [] for p in range(p_num)}
    collision_timeline = [0 for t in range(T)]
    drop_off_counter = [[] for _ in range(p_num)]
    for t in range(T):
        for p_ind in range(p_num):
            cur_s = trajs[p_ind][-1]
            next_a = np.random.choice(
                np.arange(0,A),p=flat_pols[cur_s*A:(cur_s+1)*A, t, p_ind])
            cur_sa = cur_s*A+next_a
            # print(f'player {p_ind} current state{cur_s} action {next_a}')
            
            # sometimes these transition kernels don't sum to 1 
            # just normalize them as we go
            if sum(P[p_ind][:, cur_sa]) != 1:
                P[p_ind][:, cur_sa] = P[p_ind][:, cur_sa]/sum(P[p_ind][:, cur_sa])
            next_s = np.random.choice(np.arange(0,S), p=P[p_ind][:, cur_sa])
            # print(f' next state {next_s}')
            trajs[p_ind].append(1*next_s)
            # check pick up time
            if cur_s >= S_half and next_s < S_half:
                drop_off_counter[p_ind].append(1*t)
        # collision detection
        cur_pos = [trajs[p_ind][-1] for p_ind in range(p_num)]
        for p in range(p_num):  
            collisions[p].append(cur_pos.count(cur_pos[p])-1)
            collision_timeline[t] += cur_pos.count(cur_pos[p])-1
        # collisions.append(len(cur_pos) - len(set(cur_pos)))
        # if len(cur_pos) - len(set(cur_pos)) > 0:
        #     print(f'time {t} current positions {cur_pos}')
        # collisions += 
        drop_off_time = []
        for i in range(p_num):
            drop_offs = drop_off_counter[i]
            drop_off_time.append([
                drop_offs[j+1] - drop_offs[j] for j in range(len(drop_offs)-1)])
    return collisions, collision_timeline, drop_off_time
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def value_iteration_tensor(P,c, T, minimize = True, g = 1.):
    plt.close('all')
    S, A, _ = c.shape
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    pik = np.zeros((S, T+1))
    newpi = np.zeros((S,S*A, T+1))
    total_V = np.zeros((S,T+1))
    total_V[:, -1] =  optimize(c[:, :, T], axis=1)
    pik[:, -1] = opt_arg(c[:, :, T], axis=1)
    for s in range(S):
        newpi[s, int(s*A + pik[s,-1]), -1] = 1  
    for t in range(T):
        t_ind =  T - 1 - t
        BO = c[:,:,t_ind] + g*np.einsum('ijk,i',P,total_V[:, t_ind+1])
        pik[:,t_ind] = opt_arg(BO, axis=1)
        total_V[:, t_ind] =  optimize(BO, axis=1)
        for s in range(S):
            newpi[s, int(s*A + pik[s, t_ind]), t_ind] = 1    
    return total_V, newpi

def value_iteration(P,c, minimize = True, g = 1.):
    plt.close('all')
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    # print(f' optimize is {optimize}')
    # print(f' opt_arg is {opt_arg}')
    S, A, T = c.shape
    # T = T_over - 1
    pik = np.zeros((S, T));
    newpi = np.zeros((S,S*A, T));
    Vk = np.zeros((S, T));
    BO = 1*c
    # Vk[:, T] = optimize(BO[:,:,T_over], axis=1)
    for t in range(T):
        t_ind = T - t - 1 # T - 1 , T-2, T-3, ... 0
        if t_ind  <  T - 1:
            # print(f'P shape {P.shape} vk shape {Vk[:,t_ind+1].shape}')
            BO[:,:,t_ind] +=  g*np.reshape(Vk[:,t_ind+1].dot(P), (S,A))
        # Vk[:,t_ind] = optimize(BO[:,:,t_ind], axis=1)
        pik[:,t_ind] = opt_arg(BO[:,:,t_ind], axis=1)
        # Vk[:, t_ind] = optimize(BO[:,:,t_ind], axis=1)
        for s in range(S):
            Vk[s, t_ind] = BO[s, int(pik[s, t_ind]), t_ind]
        
    for s in range(S):
        for t in range(T):
            newpi[s, int(s*A + pik[s,t]), t] = 1.

    return Vk, newpi

def value_iteration_dict(P,c, minimize = True, g = 1.):
    plt.close('all')
    optimize = np.min if minimize is True else np.max
    opt_arg = np.argmin if minimize is True else np.argmax
    S, A, T = c.shape
    # T = T_over - 1
    pik = np.zeros((S, T))
    newpi = np.zeros((S,S*A, T))
    Vk = np.zeros((S, T));
    BO = 1*c
    
    
    for t in range(T):
        t_ind = T - t - 1 # T - 1 , T-2, T-3, ... 0
        if t_ind  <  T - 1:
            # print(f'P shape {P.shape} vk shape {Vk[:,t_ind+1].shape}')
            for s in range(S):
                for a in range(A):
                    BO[s,a,t_ind] +=  g*sum([
                        prob[0]*Vk[prob[1], t_ind+1] for prob in P[(s, a)]])
        pik[:,t_ind] = opt_arg(BO[:,:,t_ind], axis=1)
        for s in range(S):
            Vk[s, t_ind] = BO[s, int(pik[s, t_ind]), t_ind]
            
    for s in range(S):
        for t in range(T):
            newpi[s, int(s*A + pik[s,t]), t] = 1.

    return Vk, newpi
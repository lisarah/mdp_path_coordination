# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:09:38 2021

@author: Sarah Li
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickWriter


def update_colors(grids, base_val, target_locs, last_locs, color_map, norm, 
                  Rows, Columns):
    for ind in range(len(last_locs)):
        last_y = (int(last_locs[ind] / Columns))
        last_x = (last_locs[ind] - last_y*Columns)   
        R,G,B,_ = color_map(norm((base_val[last_locs[ind]])))
        color = [R,G,B]              
        grids[last_y][last_x].set_facecolor(color)
        
    colors= ['xkcd:olive green', 'xkcd:bright pink'] 
    colors = [f'C{i+1}' for i in range(9)]       
    for ind in range(len(target_locs)):
        next_y = (int(target_locs[ind] / Columns))
        next_x = (target_locs[ind] - next_y*Columns) 
        grids[next_y][next_x].set_facecolor(colors[ind])

def update_cost(C, P, opponent_policy, Rows, Columns, scaler = 1.):
    C_congested = 1*C
    opponent_density = np.sum(P.dot(opponent_policy.T),axis=0)
    for y in range(Rows):
        for x in range(Columns):
            state = y*Columns + x
            valid_neighbors = []
            valid_actions = []
            if y > 0: # has up neighbor
                valid_neighbors.append((y-1)*Columns + x)
                valid_actions.append(2) # up
            if y < Rows - 1: # has bottom neighbor
                valid_neighbors.append((y+1)*Columns + x)
                valid_actions.append(3) # down
            if x > 0: # has left neighbor
                valid_neighbors.append(y*Columns + x - 1)
                valid_actions.append(0) # left
            if x< Columns - 1: # has down neighbor
                valid_neighbors.append(y*Columns + x + 1)
                valid_actions.append(1) # right
                
            for ind in range(len(valid_neighbors)):
                C_congested[state, valid_actions[ind]] += (
                    opponent_density[valid_neighbors[ind]] * scaler)
    return  C_congested

def draw_policies(Rows, Columns, policy, axis):
    length = 0.3
    lookup = {
        0: (-1, 0), # left, 
        1: (1, 0), # right, 
        2: (0, -1), #top, 
        3: (0, 1), #bottom
              }
    color = 'xkcd:coral'
    # color = 'xkcd:pale yellow'
    # color = 'xkcd:lemon'
    for x_ind in range(Columns):
        for y_ind in range(Rows):   
            # print(f'grid {x_ind}, {y_ind}')
            dx, dy = lookup[policy[y_ind*Columns + x_ind]]
            axis.arrow(x_ind + 0.5, y_ind+0.5, 
                       dx * length, dy * length, 
                       head_width=0.3, head_length=0.15, 
                       fc=color, ec=color)
    plt.show()   
def init_grid_plot(Rows, Columns, base_color):
    f, axis = plt.subplots(1)
    color_map, norm, sm = color_map_gen(base_color)
    
    value_grids = []
    for y_ind in range(Rows):
        value_grids.append([])
        for x_ind in range(Columns):
            R,G,B,_ = color_map(norm((base_color[y_ind*Columns+ x_ind])))
            color = [R,G,B]  
            value_grids[-1].append(plt.Rectangle((x_ind, -y_ind), 1, 1, 
                                                 fc=color, ec='xkcd:greyish blue'))
            axis.add_patch(value_grids[-1][-1])
    plt.axis('scaled')

    plt.colorbar(sm)
    axis.xaxis.set_visible(False)  
    axis.yaxis.set_visible(False)
    return axis, value_grids, f
def draw_policies_interpolate(Rows, Columns, p_1, p_2, axis):
    length = 0.3
    lookup = {
        0: (-1, 0), # left, 
        1: (1, 0), # right, 
        2: (0, -1), #top, 
        3: (0, 1), #bottom
              }
    color = 'xkcd:coral'
    # color = 'xkcd:pale yellow'
    color = 'xkcd:lemon'
    # halfway inbetween lemon and coral:
    color = 'xkcd:soft green'
    for x_ind in range(Columns):
        for y_ind in range(Rows):   
            # print(f'grid {x_ind}, {y_ind}')
            dx_1, dy_1 = lookup[p_1[y_ind*Columns + x_ind]]
            dx_2, dy_2 = lookup[p_2[y_ind*Columns + x_ind]]
            axis.arrow(x_ind + 0.5, y_ind+0.5, 
                       (dx_1 + dx_2) * 0.5 * length, 
                       (dy_1 + dy_2) * 0.5 * length, 
                       head_width=0.3, head_length=0.15, 
                       fc=color, ec=color)
    plt.show()  
    
    
# def init_grid_plot(Rows, Columns, base_color):
#     color_map, norm, _ = color_map_gen(base_color)
#     f, axis = plt.subplots(1)
#     value_grids = []
#     for y_ind in range(Rows):
#         value_grids.append([])
#         for x_ind in range(Columns):
#             R,G,B,_ = color_map(norm((base_color[y_ind*Columns+ x_ind])))
#             color = [R,G,B]  
#             value_grids[-1].append(plt.Rectangle((x_ind, y_ind), 1, 1, 
#                                                  fc=color, ec='xkcd:greyish blue'))
#             axis.add_patch(value_grids[-1][-1])
#     plt.axis('scaled')
#     axis.xaxis.set_visible(False)  
#     axis.yaxis.set_visible(False)
#     return axis, value_grids

def simulate(p1_init, p2_init, policies, base_color, value_grids, 
             A, Rows, Columns, P, Time = 100):
    p1_traj = [p1_init]
    p2_traj = [p2_init]
    
    color_map, norm, _ = color_map_gen(base_color)
    
    for t in range(Time):
        flat_policy = np.sum(policies[:,:,t,:], axis=0)
        state_1 = p1_traj[-1]
        state_2 = p2_traj[-1]
        # print(flat_policy[state*A:(state+1)*A, 0])
        next_action_1 = np.random.choice(
            np.arange(0,A),p=flat_policy[state_1*A:(state_1+1)*A, 0])
        next_action_2 = np.random.choice(
            np.arange(0,A),p=flat_policy[state_2*A:(state_2+1)*A, 1])
        # print(P[:, state*A+next_action])
        next_state_1 = np.random.choice(np.arange(0,Rows*Columns), 
                                      p=P[:, state_1*A+next_action_1])
        next_state_2 = np.random.choice(np.arange(0,Rows*Columns), 
                                      p=P[:, state_2*A+next_action_2])
        p1_traj.append(next_state_1)
        p2_traj.append(next_state_2)
        update_colors(value_grids, base_color, [next_state_1, next_state_2], 
                      [state_1, state_2], color_map, norm, Rows, Columns)
        plt.pause(1)
        plt.show()
        
    plt.show()
    
def color_map_gen(base_color):
    v_max = np.max(base_color)
    v_min = np.min(base_color)
    print(f'v_min = {v_min}')
    print(f'v_max = {v_max}')
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    color_map = plt.get_cmap('coolwarm')  
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])
    return color_map, norm, sm

def animate_traj(file_name, f, p_inits, policies, base_color, value_grids, 
             A, Rows, Columns, P, Time = 100):
    p_iter = range(len(p_inits))
    
    p_trajs  = [[p_inits[p]] for p in p_iter]
    S, _ = P[0].shape
    color_map, norm, _ = color_map_gen(base_color)
    white_color = np.zeros(base_color.shape)
    for t in range(Time):
        flat_policy = np.sum(policies[:,:,t,:], axis=0)
        for p_ind in p_iter:
            cur_s = p_trajs[p_ind][-1] # pick up mode
            next_a = np.random.choice(
                # np.arange(0,A),p=policies[cur_s*A:(cur_s+1)*A, p_ind])
                np.arange(0,A),p=flat_policy[cur_s*A:(cur_s+1)*A, p_ind])
            
            next_s = np.random.choice(np.arange(0,S), 
                                          p=P[p_ind][:, cur_s*A+next_a])
            p_trajs[p_ind].append(next_s)
        
    def animate(i):
        if i == 0:
            states = [p_trajs[p][0] %(Rows*Columns) for p in p_iter]
            next_states = states
            # state_1 = p1_traj[0]
            # state_2 = p2_traj[0]
        else:
            states = [p_trajs[p][i-1] %(Rows*Columns) for p in p_iter]
            next_states  = [p_trajs[p][i] %(Rows*Columns) for p in p_iter]
            # state_1 = p1_traj[i-1]
            # state_2 = p2_traj[i-1]
                
        for p_ind in range(len(states)):
            if states[p_ind] >= Rows*Columns:
                if states[p_ind] <  Rows*Columns + Columns: # top row
                    print(f'{p_ind} in pick up mode')
                states[p_ind] = states[p_ind] - Rows*Columns
            elif states[p_ind] >= Rows*Columns - Columns:  # bottom row
                print(f'{p_ind} in delivery mode')
        update_colors(value_grids, white_color, next_states, states, 
                      color_map, norm, Rows, Columns)
  
    ani = animation.FuncAnimation(f, animate, frames=range(Time), interval=250)
    plt.show()
    ani.save(file_name, writer='ffmpeg')  # imagemagick
    
    
    
    
    
    
    
    
    
    
    
    
    
    
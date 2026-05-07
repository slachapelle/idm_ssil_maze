import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mazelab import VonNeumannMotion
from mazelab import MooreMotion


def xy_to_flatten_idx(array, x, y):
    M, N = array.shape
    return x*N + y


def flatten_idx_to_xy(array, idx):
    M, N = array.shape
    x = idx//N
    y = idx%N
    return np.array([x, y])


def make_graph(impassable_array, motions):
    M, N = impassable_array.shape
    free_idx = np.stack(np.where(np.logical_not(impassable_array)), axis=1)
    row = []
    col = []
    for idx in free_idx:
        node_idx = xy_to_flatten_idx(impassable_array, idx[0], idx[1])
        for motion in motions:
            next_idx = [idx[0] + motion[0], idx[1] + motion[1]]
            if (next_idx[0] >= 0 and next_idx[0] < M and next_idx[1] >= 0 and next_idx[1] < N) and not impassable_array[next_idx[0], next_idx[1]]:
                row.append(node_idx)
                col.append(xy_to_flatten_idx(impassable_array, next_idx[0], next_idx[1]))
    data = [1]*len(row)
    graph = csr_matrix((data, (row, col)), shape=(M*N, M*N))
    
    return graph


def get_actions(impassable_array, motions, predecessors, start_idx, goal_idx):
    start_idx = xy_to_flatten_idx(impassable_array, *start_idx)
    goal_idx = xy_to_flatten_idx(impassable_array, *goal_idx)
    actions = []
    while goal_idx != start_idx:
        if predecessors[goal_idx] == -9999:
            return None
        action = flatten_idx_to_xy(impassable_array, goal_idx) - flatten_idx_to_xy(impassable_array, predecessors[goal_idx])
        for i, motion in enumerate(motions):
            if np.allclose(action, motion):
                action_idx = i
        actions.append(action_idx)
        goal_idx = predecessors[goal_idx]
    return actions[::-1]


def dijkstra_solver(impassable_array, motions, start_idx, goal_idx):
    impassable_array = np.asarray(impassable_array)
    assert impassable_array.dtype == bool
    assert isinstance(motions, (VonNeumannMotion, MooreMotion))
    
    graph = make_graph(impassable_array, motions)
    dist_matrix, predecessors = dijkstra(csgraph=graph, indices=xy_to_flatten_idx(impassable_array, *start_idx), return_predecessors=True)
    actions = get_actions(impassable_array, motions, predecessors, start_idx, goal_idx)
    return actions

def get_policy(impassable_array, motions, goal_xy):
    goal_idx = xy_to_flatten_idx(impassable_array, *goal_xy)
    impassable_array = np.asarray(impassable_array)
    maze_shape = impassable_array.shape
    assert impassable_array.dtype == bool
    assert isinstance(motions, (VonNeumannMotion, MooreMotion))
    
    graph = make_graph(impassable_array, motions)
    _, next_idx = dijkstra(csgraph=graph, indices=goal_idx, return_predecessors=True)

    n_nodes = len(next_idx)
    policy = np.zeros(maze_shape + (2,), dtype=np.int32)
    policy_idx = np.zeros(maze_shape, dtype=np.int32)
    for idx in range(n_nodes):
        xy_node = flatten_idx_to_xy(impassable_array, idx)
        next_xy_node = flatten_idx_to_xy(impassable_array, next_idx[idx])
        if next_idx[idx] == -9999:
            #action = [-9,-9]  # special null value
            action = [0, 0]
        else:
            action = next_xy_node - xy_node 
        policy[xy_node[0], xy_node[1], :] = action
    
        if next_idx[idx] == -9999:
            #policy_idx[xy_node[0], xy_node[1]] = -9  # special null value
            policy_idx[xy_node[0], xy_node[1]] = -1  # special null value
        else:
            found = False
            for i, motion in enumerate(motions):
                if np.allclose(action, motion):
                    found = True
                    policy_idx[xy_node[0], xy_node[1]] = i
            assert found
            
    return policy, policy_idx


import enum
import os

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import cv2
from einops import rearrange

markers = ('^', '>', 'v', '<')  
colors = ('red', 'blue', 'green', 'purple')

def torch_predict(model, X):
    device = next(model.parameters()).device
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds


class Direction(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

sebidx2sachaidx = {0:0, 1:2, 2:3, 3:1}
sachidx2move= {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}
#markers = ('^', '>', 'v', '<')


def get_base_maze_from_policy(policy, maze_shape, goal):
    # represent goal in xy coord
    goal = [goal[1], (maze_shape[0] - 1) - goal[0]]

    # Create base maze in x y coord
    base_maze = np.ones((3, maze_shape[1], maze_shape[0]))
    for i in range(maze_shape[0]):
        for j in range(maze_shape[1]):
            if policy[i,j] == -1:  # means the policy is not defined there, which means there's a wall
                base_maze[:, j, (maze_shape[0] - 1) - i] = np.array([0.0, 0.0, 1.0])
    base_maze[:, goal[0], goal[1]] = np.array([0.0, 1.0, 0.0])  # add goal
    return base_maze

def build_dataset_from_policy(policy, num_inter=10):
    maze_shape = policy.shape
    x_policy = []
    y_policy = []
    x_idm = []

    for i in range(maze_shape[0]):
        for j in range(maze_shape[1]):
            if policy[i,j] != -1:
                #current_state = [i, j]
                #current_state = [j, -i]  # x y representation
                current_state = [j, (maze_shape[0] - 1) - i]  # x y representation
                action_idx = sebidx2sachaidx[policy[i,j]]
                action_move = sachidx2move[action_idx]
                for _ in range(num_inter):  # Making the state space denser by interpolating (1 action becomes num_inter actions)
                    next_state = [current_state[0] + action_move[0] / num_inter, current_state[1] + action_move[1] / num_inter]
                    x_policy.append(current_state)
                    y_policy.append(action_idx)
                    x_idm.append(current_state + next_state)
                    current_state = next_state

    x_policy = np.array(x_policy)
    y_policy = np.array(y_policy)
    x_idm = np.array(x_idm)

    return x_policy, y_policy, x_idm, y_policy  # y_idm == y_policy


def build_dataset_from_policy_stoch_env_v1(policy, goal, prob=0.1, num_inter=1, samples_mult=1):
    # prob = probability of taking two steps instead of one step.
    assert num_inter == 1
    maze_shape = policy.shape
    x_policy = []
    y_policy = []
    x_idm = []

    # Create base maze in x y coord
    base_maze = get_base_maze_from_policy(policy, maze_shape, goal)
    for _ in range(samples_mult):
        for i in range(maze_shape[0]):
            for j in range(maze_shape[1]):
                if policy[i,j] != -1:  # where policy is defined, i.e. where no wall.
                    #current_state = [i, j]
                    #current_state = [j, -i]  # x y representation
                    current_state = [j, (maze_shape[0] - 1) - i]  # x y representation
                    action_idx = sebidx2sachaidx[policy[i,j]]
                    action_move = sachidx2move[action_idx]
                    num_steps = np.where(np.random.random() < prob, 2, 1)
                    #import ipdb; ipdb.set_trace()
                    next_state = [current_state[0] + num_steps * action_move[0], current_state[1] + num_steps * action_move[1]]
                    # check if this move isn't allowed, take only one step
                    if np.all(base_maze[:, next_state[0], next_state[1]] == np.array([0.0, 0.0, 1.0])):
                        #print("found unallowed move, changing next_state accordingly.")
                        assert num_steps == 2, "if num_steps == 1, the move should be allowed"
                        # move by only one step, which should be allowed
                        next_state = [current_state[0] + action_move[0], current_state[1] + action_move[1]]
                    x_policy.append(current_state)
                    y_policy.append(action_idx)
                    x_idm.append(current_state + next_state)

    x_policy = np.array(x_policy)
    y_policy = np.array(y_policy)
    x_idm = np.array(x_idm)

    return x_policy, y_policy, x_idm, y_policy  # y_idm == y_policy


def build_dataset_from_policy_stoch_env_v2(policy, prob=0.1, samples_mult=1):
    maze_shape = policy.shape
    x_policy = []
    y_policy = []
    x_idm = []

    for _ in range(samples_mult):
        for i in range(maze_shape[0]):
            for j in range(maze_shape[1]):
                if policy[i,j] != -1:  # where policy is defined, i.e. where no wall.
                    #current_state = [i, j]
                    #current_state = [j, -i]  # x y representation
                    current_state = [j, (maze_shape[0] - 1) - i]  # x y representation
                    action_idx = sebidx2sachaidx[policy[i,j]]
                    action_move = sachidx2move[action_idx]
                    if np.random.random() < prob:
                        # action does nothing
                        next_state = current_state
                    else:
                        # action is executed normally
                        next_state = [current_state[0] + action_move[0], current_state[1] + action_move[1]]
                    x_policy.append(current_state)
                    y_policy.append(action_idx)
                    x_idm.append(current_state + next_state)

    x_policy = np.array(x_policy)
    y_policy = np.array(y_policy)
    x_idm = np.array(x_idm)

    return x_policy, y_policy, x_idm, y_policy  # y_idm == y_policy


def build_img_dataset_from_policy(policy, goal):
    maze_shape = policy.shape
    x_policy = []
    y_policy = []
    x_idm = []

    # Create base maze in x y coord
    base_maze = get_base_maze_from_policy(policy, maze_shape, goal)
    
    def state_to_img(state):
        img = np.zeros((3, maze_shape[1], maze_shape[0]))
        img[:,:,:] = base_maze
        img[:, state[0], state[1]] = np.array([1.0, 0.0, 0.0])
        return img

    for i in range(maze_shape[0]):
        for j in range(maze_shape[1]):
            if policy[i,j] != -1:
                #current_state = [i, j]
                current_state = [j, (maze_shape[0] - 1) - i]  # x y representation
                action_idx = sebidx2sachaidx[policy[i,j]]
                action_move = sachidx2move[action_idx]
                
                next_state = [current_state[0] + action_move[0], current_state[1] + action_move[1]]
                current_img = state_to_img(current_state)
                next_img = state_to_img(next_state)

                x_policy.append(current_img)
                y_policy.append(action_idx)
                x_idm.append(np.concatenate([current_img, next_img], 0))
    
    x_policy = np.array(x_policy)
    y_policy = np.array(y_policy)
    x_idm = np.array(x_idm)

    return x_policy, y_policy, x_idm, y_policy  # y_idm == y_policy


def load_data(folder, num_inter=10, num_goals=1, add_goal=False):
    x_policy_all, x_idm_all, y_policy_all, y_idm_all = [], [], [], []
    num_goals_ = len([f for f in os.listdir(folder) if f.endswith("goal.pkl")])
    assert num_goals <= num_goals_
    for g in range(num_goals):
        policy = np.load(os.path.join(folder, f"g{g}_policy_idx.npy"))
        x_policy, y_policy, x_idm, y_idm = build_dataset_from_policy(policy, num_inter=num_inter)
        if add_goal:
            with open(os.path.join(folder, f"g{g}_goal.pkl"), "rb") as f:
                goal = pickle.load(f)
                goal = np.array([goal[1], -goal[0]])  # setting goal in xy coordinates (originally in ij)
            # add goal information
            goal_expanded = np.broadcast_to(goal, (x_policy.shape[0], goal.shape[0]))
            x_policy = np.concatenate([x_policy, goal_expanded], axis=1)
            x_idm = np.concatenate([x_idm, goal_expanded], axis=1)

        x_policy_all.append(x_policy)
        x_idm_all.append(x_idm)
        y_policy_all.append(y_policy)
        y_idm_all.append(y_idm)
    
    x_policy_all = np.concatenate(x_policy_all, axis=0)
    x_idm_all = np.concatenate(x_idm_all, axis=0)
    y_policy_all = np.concatenate(y_policy_all, axis=0)
    y_idm_all = np.concatenate(y_idm_all, axis=0)
    return x_policy_all, y_policy_all, x_idm_all, y_idm_all

def load_data_stoch_env(folder, prob=0.0, samples_mult=1, env_version='v1'):
    policy = np.load(os.path.join(folder, f"g0_policy_idx.npy"))
    with open(os.path.join(folder, f"g0_goal.pkl"), "rb") as f:
        goal = pickle.load(f)
    if env_version == 'v1':
        x_policy, y_policy, x_idm, y_idm = build_dataset_from_policy_stoch_env_v1(policy, goal, prob=prob, samples_mult=samples_mult)
    elif env_version == 'v2':
        x_policy, y_policy, x_idm, y_idm = build_dataset_from_policy_stoch_env_v2(policy, prob=prob, samples_mult=samples_mult)
    else:
        raise NotImplementedError(f"Environment {env_version} does not exist.")

    return x_policy, y_policy, x_idm, y_idm

def load_img_data(folder, num_goals=1, add_goal=False, noise_std=0, noise_mult=1):
    x_policy_all, x_idm_all, y_policy_all, y_idm_all = [], [], [], []
    num_goals_ = len([f for f in os.listdir(folder) if f.endswith("goal.pkl")])
    assert num_goals <= num_goals_
    for g in range(num_goals):
        policy = np.load(os.path.join(folder, f"g{g}_policy_idx.npy"))
        with open(os.path.join(folder, f"g{g}_goal.pkl"), "rb") as f:
                goal = pickle.load(f)
        x_policy, y_policy, x_idm, y_idm = build_img_dataset_from_policy(policy, goal)

        if add_goal:
            # add goal information
            goal = np.array([goal[1], -goal[0]])  # setting goal in xy coordinates (originally in ij)
            goal_expanded = np.broadcast_to(goal, (x_policy.shape[0], goal.shape[0]))
            x_policy = np.concatenate([x_policy, goal_expanded], axis=1)
            x_idm = np.concatenate([x_idm, goal_expanded], axis=1)

        x_policy_all.append(x_policy)
        x_idm_all.append(x_idm)
        y_policy_all.append(y_policy)
        y_idm_all.append(y_idm)
    
    x_policy_all = np.concatenate(x_policy_all, axis=0)
    x_idm_all = np.concatenate(x_idm_all, axis=0)
    y_policy_all = np.concatenate(y_policy_all, axis=0)
    y_idm_all = np.concatenate(y_idm_all, axis=0)

    if noise_std > 0:
        shape = x_policy_all.shape
        noise = np.random.normal(0, noise_std, size=(noise_mult,) + shape)
        x_policy_all = rearrange((x_policy_all[np.newaxis, ...] + noise), "m b ... -> (m b) ...")
        y_policy_all = rearrange((y_policy_all[np.newaxis, ...] + np.zeros((noise_mult,) + y_policy_all.shape)), "m b ... -> (m b) ...")

        shape = x_idm_all.shape
        noise = np.random.normal(0, noise_std, size=(noise_mult,) + shape)
        x_idm_all = rearrange((x_idm_all[np.newaxis, ...] + noise), "m b ... -> (m b) ...")
        y_idm_all = rearrange((y_idm_all[np.newaxis, ...] + np.zeros((noise_mult,) + y_idm_all.shape)), "m b ... -> (m b) ...")

        x_policy_all = np.clip(x_policy_all, 0, 1)
        x_idm_all = np.clip(x_idm_all, 0, 1)

    return x_policy_all, y_policy_all, x_idm_all, y_idm_all


def load_maze_img(folder):
    # Load the video file
    video_path = f'{folder}/maze_g0_s0.mp4'  # Replace with your actual file path
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Failed to read the video file.")

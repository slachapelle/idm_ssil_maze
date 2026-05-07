from typing import Tuple, List, Optional
import os
import pickle
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse

from training import train_torch_model, test_loss_eval
from utils import torch_predict
from plot import avg_rew_barplot
from models import TorchLogistic, TorchMLP

# /home/mila/l/lachaseb/mazelab/train/learn_maze_many_sol.py

# Actions: 0=left, 1=right, 2=down, 3=up
ACTION_NAMES = ("left", "right", "down", "up")

class GridEnv:
    """
    Square grid environment.
    State is (row, col) with (0,0) top-left and (n-1,n-1) bottom-right goal.
    Deterministic transitions. No interior walls. Trying to move outside keeps you in place.
    """

    def __init__(self, size: int):
        assert size >= 2, "size must be >= 2"
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start

    def evaluate_policy(self, model, num_episodes: int, device) -> float:
        """
        Evaluate the policy model on the environment.
        Returns the proportion of episodes that reach the goal.
        """
        model.eval()
        successes = 0
        distance_sum = 0
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            for _ in range(num_episodes):
                state = self.reset()
                #import ipdb; ipdb.set_trace()
                max_steps = int(2.0 * (self.size-1))
                best_distance = np.inf
                for _ in range(max_steps):
                    # scale state
                    scaled_state = model.scaler.transform(np.array([state]))
                    # Convert state to tensor and get action
                    state_tensor = torch.tensor(scaled_state, dtype=torch.float32).to(device)
                    logits = model(state_tensor)
                    action = logits.argmax(dim=1).item()
                    #print(f"state {state} action {action} ({ACTION_NAMES[action]}) logits {logits}")
                    # Step environment
                    state, _, done = self.step(action)
                    distance = np.abs(state[0] - self.goal[0]) + np.abs(state[1] - self.goal[1])
                    if distance < best_distance:
                        best_distance = distance
                    if done:
                        successes += 1
                        break
                distance_sum += best_distance
        return successes / num_episodes, distance_sum / num_episodes

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        r, c = self.state
        if action == 0:  # left
            nc = max(0, c - 1)
            nr = r
        elif action == 1:  # right
            nc = min(self.size - 1, c + 1)
            nr = r
        elif action == 2:  # down
            nr = min(self.size - 1, r + 1)
            nc = c
        elif action == 3:  # up
            nr = max(0, r - 1)
            nc = c
        else:
            raise ValueError("invalid action")

        self.state = (nr, nc)
        done = self.state == self.goal
        reward = 1.0 if done else 0.0
        return self.state, reward, done


def expert_policy_stoch(state, size: int, prob=0.5) -> Optional[np.ndarray]:
    """
    Expert policy:
    - On right border (col == size-1): choose down with prob 1.
    - On bottom border (row == size-1): choose right with prob 1.
    - Else: choose right or down with probability 1/2 each.
    - At goal: return None (single state) or zero row (batched).
    Returns a length-4 numpy array of probabilities over actions [left,right,down,up],
    or shape (N, 4) when state is an array of shape (N, 2).
    """
    # Actions: 0=left, 1=right, 2=down, 3=up
    state_arr = np.asarray(state)
    batched = state_arr.ndim == 2
    if not batched:
        state_arr = state_arr[None, :]

    # Batched: state_arr shape (N, 2)
    r, c = state_arr[:, 0], state_arr[:, 1]
    N = len(r)
    probs = np.zeros((N, 4), dtype=float)
    on_right   = (c == size - 1) & (r != size - 1)
    on_bottom  = (r == size - 1) & (c != size - 1)
    interior   = (r != size - 1) & (c != size - 1)
    probs[on_right,  2] = 1.0
    probs[on_bottom, 1] = 1.0
    probs[interior,  1] = prob
    probs[interior,  2] = 1 - prob
    if N == 1:
        probs = probs.reshape((4,))
    return probs

def compute_policy_star_entropy(X_int: np.ndarray, env_size: int, expert_policy) -> float:
    """
    Compute E_{p*(x)}[H(π*(·|x))] empirically using states in X_int.
    X_int: shape (N, 2), integer (row, col) coordinates from the DGP.
    Calls expert_policy to get the true action distribution at each state,
    then averages the Shannon entropy over the dataset.
    """
    probs = expert_policy(X_int, env_size)  # (N, 4)
    log_probs = np.where(probs > 0, np.log(probs), 0.0)  # 0*log0 = 0
    return float(np.mean(-np.sum(probs * log_probs, axis=1)))


def sample_trajectory(env: GridEnv, policy, rng: np.random.Generator, max_steps: Optional[int] = None) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Sample a single trajectory (states and actions) from the expert policy until the goal is reached.
    Returns lists: states (pre-action states) and actions taken at those states.
    """
    if max_steps is None:
        max_steps = 4 * env.size  # safe upper bound (actual needed is 2*size-2)

    states = []
    actions = []
    state = env.reset()
    for _ in range(max_steps):
        probs = policy(state, env.size)
        if probs.sum() == 0:
            break
        action = rng.choice(len(probs), p=probs)
        states.append(state)
        actions.append(int(action))
        state, _, done = env.step(action)
        if done:
            states.append(state)
            break
    return states, actions


def generate_dataset(env, policy, n_trajectories: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset by sampling n_trajectories independent rollouts from the expert.
    Returns X, y:
      - X is an array of shape (N, 2) with rows being (row, col) coordinates.
      - y is an array of shape (N,) with integer actions {0,1,2,3}.
    """
    rng = np.random.default_rng(seed)
    X_list = []
    X_next_list = []
    y_list = []
    for _ in range(n_trajectories):
        states, actions = sample_trajectory(env, policy, rng)
        X_list.extend(states[:-1])
        X_next_list.extend(states[1:])
        y_list.extend(actions)
    X_policy = np.array(X_list, dtype=int).reshape(-1, 2)
    X_next = np.array(X_next_list, dtype=int).reshape(-1, 2)
    X_idm = np.concatenate([X_policy, X_next], 1)
    y = np.array(y_list, dtype=int).reshape(-1)

    # shuffle dataset rows consistently
    perm = rng.permutation(X_policy.shape[0])
    X_policy = X_policy[perm]
    X_idm = X_idm[perm]
    y = y[perm]

    return X_policy, X_idm, y


def plot_learning_curves(data_plots, output_folder, model_name=None):
    import matplotlib.pyplot as plt

    # Plotting loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data_plots['train_loss']['curve'], label='Train Loss')
    plt.plot(data_plots['test_loss']['curve'], label='Test Loss')
    plt.title(f'Loss Curves ({model_name})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(data_plots['train_acc']['curve'], label='Train Accuracy')
    plt.plot(data_plots['test_acc']['curve'], label='Test Accuracy')
    plt.title(f'Accuracy Curves ({model_name})')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'learning_curves_{model_name}.png'))
    plt.show()


def plot_train_sample_heatmap(X_train, env_size, save_path=None, annotate=False):
    import matplotlib.pyplot as plt
    counts = np.zeros((env_size, env_size), dtype=int)
    # X_train rows are (row, col)
    for r, c in X_train:
        counts[int(r), int(c)] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(counts, cmap="viridis", origin="upper")
    ax.set_title("Training samples per grid cell")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_xticks(np.arange(env_size))
    ax.set_yticks(np.arange(env_size))
    plt.colorbar(im, ax=ax, label="count")

    if annotate:
        for i in range(env_size):
            for j in range(env_size):
                text = ax.text(j, i, counts[i, j], ha="center", va="center", color="w" if counts[i, j] > counts.max()/2 else "k", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn maze policy with PyTorch.")
    parser.add_argument("--model", type=str, choices=["LOGISTIC", "MLP"], default="MLP", help="Model type: LOGISTIC or MLP")
    parser.add_argument("--max_iter", type=float, default=20000, help="max number of iterations. default is infinite.")
    parser.add_argument("--meta_random_state", type=int, default=9988338, help="Meta random seed")
    parser.add_argument("--env_size", type=int, default=20, help="Size of the env grid")
    parser.add_argument("--expert_prob", type=float, default=0.5, help="stochasticity of the expert")
    #parser.add_argument("--num_samples_test", type=int, default=10*100**2, help="Num samples total")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--output_folder", type=str, default="", help="Where outputs will be saved")

    args = parser.parse_args()
    args.output_folder = os.path.join(args.output_folder, f"{args.model}_{args.env_size}env_{args.expert_prob}p_{args.num_samples}n_{args.num_seeds}seeds")  # where to save the plots

    args.train_splits = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    np.random.seed(args.meta_random_state)
    RANDOM_STATES = np.random.randint(0, high=1000000, size=args.num_seeds, dtype=int).tolist()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    env = GridEnv(args.env_size)
    
    data_plots = {}
    last_plots = {}

    for TRAIN_SPLIT in args.train_splits:
        data_plots[TRAIN_SPLIT] = {}
        last_plots[TRAIN_SPLIT] = {}
        last_plots[TRAIN_SPLIT]["policy"] = dict(train_loss=[], train_acc=[], avg_rew=[], avg_dist=[], test_loss=[], kl=[])
        last_plots[TRAIN_SPLIT]["idm"] = dict(train_loss=[], train_acc=[], test_loss=[], kl=[])
        last_plots[TRAIN_SPLIT]['idm_policy'] = dict(train_loss=[], train_acc=[], avg_rew=[], avg_dist=[], test_loss=[], kl=[])
        if not os.path.exists(f"{args.output_folder}/split_{TRAIN_SPLIT}"):
            os.makedirs(f"{args.output_folder}/split_{TRAIN_SPLIT}")
    
        for RANDOM_STATE in RANDOM_STATES:
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)

            seed_folder = f"{args.output_folder}/split_{TRAIN_SPLIT}/seed{RANDOM_STATE}"
            if not os.path.exists(seed_folder):
                os.makedirs(seed_folder)

            # Randomly sample data (seeded)
            args.num_samples
            n_trajectories = int(args.num_samples / (2 * args.env_size - 2))
            expert_policy = lambda state, size : expert_policy_stoch(state, size, args.expert_prob)
            X_policy, X_idm, y = generate_dataset(env=env, policy=expert_policy, n_trajectories=n_trajectories, seed=RANDOM_STATE)
            num_samples = X_policy.shape[0]

            # Fixed-size test set: independent of TRAIN_SPLIT, generated with a separate seed
            X_policy_test_raw, X_idm_test_raw, y_test = generate_dataset(
                env=env, policy=expert_policy, n_trajectories=10*n_trajectories, seed=RANDOM_STATE + 10**6
            )

            # shuffle data
            perm = np.random.choice(range(num_samples), size=num_samples, replace=False)
            X_policy = X_policy[perm]
            X_idm = X_idm[perm]
            y = y[perm]
            
            # select labeled samples
            num_samples_labeled = int(num_samples * TRAIN_SPLIT)
            X_policy_lab = X_policy[:num_samples_labeled]
            X_idm_lab = X_idm[:num_samples_labeled]
            y = y[:num_samples_labeled]  # throwing out bunch of labels

            plot_train_sample_heatmap(X_policy_lab, args.env_size, seed_folder)

            # Modeling
            if args.model == "LOGISTIC":
                policy_model = TorchLogistic(X_policy.shape[1], 4)
                idm_model = TorchLogistic(X_idm.shape[1], 4)
                idm_policy_model = TorchLogistic(X_policy.shape[1], 4)
            elif args.model == "MLP":
                model_args = dict(hidden_dims=(100, 100, 100, 100, 100))
                policy_model = TorchMLP(X_policy.shape[1], 4, **model_args)
                idm_model = TorchMLP(X_idm.shape[1], 4, **model_args)
                idm_policy_model = TorchMLP(X_policy.shape[1], 4, **model_args)

            else:
                raise ValueError("Unknown MODE: {}".format(args.model))

            X_policy = policy_model.scaler.fit_transform(X_policy)
            X_policy_lab = policy_model.scaler.transform(X_policy_lab)
            X_policy_test = policy_model.scaler.transform(X_policy_test_raw.astype(float))

            X_idm = idm_model.scaler.fit_transform(X_idm)
            X_idm_lab = idm_model.scaler.transform(X_idm_lab)
            X_idm_test = idm_model.scaler.transform(X_idm_test_raw.astype(float))

            idm_policy_model.scaler = policy_model.scaler

            print("BC...")
            _t0 = time.time()
            policy_model, data_plots[TRAIN_SPLIT]["policy"] = train_torch_model(policy_model, X_policy_lab, y, max_iter=args.max_iter, batch_size=512, env=None)
            print(f"   done in {time.time() - _t0:.2f}s")

            print("IDM labeling...")
            print("1- Train IDM")
            _t0 = time.time()
            idm_model, data_plots[TRAIN_SPLIT]["idm"] = train_torch_model(idm_model, X_idm_lab, y, max_iter=args.max_iter, env=None)  # no env, since idm model can't be used for rollouts
            print(f"   done in {time.time() - _t0:.2f}s")
            print("2- Label dataset")
            _t0 = time.time()
            y_idm = torch_predict(idm_model, X_idm)
            print(f"   done in {time.time() - _t0:.2f}s")
            print("3- Train IDM based policy")
            _t0 = time.time()
            idm_policy_model, data_plots[TRAIN_SPLIT]["idm_policy"] = train_torch_model(idm_policy_model, X_policy, y_idm, max_iter=args.max_iter, batch_size=512, env=None)
            print(f"   done in {time.time() - _t0:.2f}s")

            print("Final eval...")
            _t0 = time.time()
            # Estimate KL divergences on the fixed test set
            policy_ce = test_loss_eval(policy_model, X_policy_test, y_test, batch_size=1024)
            idm_ce = test_loss_eval(idm_model, X_idm_test, y_test, batch_size=1024)
            idm_policy_ce = test_loss_eval(idm_policy_model, X_policy_test, y_test, batch_size=1024)
            policy_star_entropy = compute_policy_star_entropy(X_policy_test_raw, args.env_size, expert_policy)
            # KL(π*||π) = cross-entropy(π*, π) − H(π*)
            kl_policy = policy_ce - policy_star_entropy
            kl_idm_policy = idm_policy_ce - policy_star_entropy
            # KL(h*||h) = cross-entropy − H(h*) = CE − 0  (h* is deterministic)
            kl_idm = idm_ce

            # Evaluate policy
            device = next(policy_model.parameters()).device
            policy_avg_rew, policy_avg_dist = env.evaluate_policy(policy_model, num_episodes=1000, device=device)
            idm_policy_avg_rew, idm_policy_avg_dist = env.evaluate_policy(idm_policy_model, num_episodes=1000, device=device)
            print(f"   done in {time.time() - _t0:.2f}s")

            # record last performance of the curve
            _derived = {"kl", "test_loss", 'avg_rew', 'avg_dist'}
            for key in last_plots[TRAIN_SPLIT]["policy"].keys():
                if key not in _derived:
                    last_plots[TRAIN_SPLIT]["policy"][key].append(data_plots[TRAIN_SPLIT]["policy"][key]["curve"][-1])
            for key in last_plots[TRAIN_SPLIT]["idm"].keys():
                if key not in _derived:
                    last_plots[TRAIN_SPLIT]["idm"][key].append(data_plots[TRAIN_SPLIT]['idm'][key]["curve"][-1])
            for key in last_plots[TRAIN_SPLIT]["idm_policy"].keys():
                if key not in _derived:
                    last_plots[TRAIN_SPLIT]["idm_policy"][key].append(data_plots[TRAIN_SPLIT]['idm_policy'][key]["curve"][-1])
            last_plots[TRAIN_SPLIT]["policy"]["kl"].append(kl_policy)
            last_plots[TRAIN_SPLIT]["idm"]["kl"].append(kl_idm)
            last_plots[TRAIN_SPLIT]["idm_policy"]["kl"].append(kl_idm_policy)
            last_plots[TRAIN_SPLIT]["policy"]["test_loss"].append(policy_ce)
            last_plots[TRAIN_SPLIT]["idm"]["test_loss"].append(idm_ce)
            last_plots[TRAIN_SPLIT]["idm_policy"]["test_loss"].append(idm_policy_ce)
            last_plots[TRAIN_SPLIT]["policy"]["avg_rew"].append(policy_avg_rew)
            last_plots[TRAIN_SPLIT]["policy"]["avg_dist"].append(policy_avg_dist)
            last_plots[TRAIN_SPLIT]["idm_policy"]["avg_rew"].append(idm_policy_avg_rew)
            last_plots[TRAIN_SPLIT]["idm_policy"]["avg_dist"].append(idm_policy_avg_dist)
            
            #plot_learning_curves(data_plots["policy"], seed_folder, model_name="policy")
            #plot_learning_curves(data_plots["idm"], seed_folder, model_name="idm")
        
            with open(f"{seed_folder}/data_plots.pkl", "wb") as f:
                pickle.dump(data_plots[TRAIN_SPLIT], f)
            torch.save(policy_model, f"{seed_folder}/policy_model.pt")
            torch.save(idm_model, f"{seed_folder}/idm_model.pt")
            torch.save(idm_policy_model, f"{seed_folder}/idm_policy_model.pt")
            
    
    # plot avg_reward
    #avg_rew_barplot(last_plots, args)
    print(last_plots)
    with open(f"{args.output_folder}/last_plots.pkl", "wb") as f:
        pickle.dump(last_plots, f)
    with open(f"{args.output_folder}/args.pkl", "wb") as f:
        pickle.dump(args, f)

    

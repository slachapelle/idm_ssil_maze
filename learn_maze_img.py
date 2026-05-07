import os
import enum
import pickle
import math

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse

from plot import aggregated_plot, plot_acc_curves, plot_loss_curves, plot_labels
from utils import torch_predict, load_img_data
from training import train_torch_model
from models import TorchLogistic, TorchMLP, TorchCNN, TorchLogisticCNN
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn maze policy with PyTorch.")
    parser.add_argument("--model", type=str, choices=["LOGISTIC", "MLP", "LINEAR_CNN", "CNN"], default="MLP", help="Model type: LOGISTIC or MLP")
    parser.add_argument("--max_iter", type=float, default=200000, help="max number of iterations. default is infinite.")
    parser.add_argument("--add_goal", action="store_true", default=False, help="Whether to add goal information (default: False)")
    parser.add_argument("--meta_random_state", type=int, default=9988338, help="Meta random seed")
    parser.add_argument("--maze", type=str, default="20x20", help="Maze name")
    parser.add_argument("--noise_std", type=float, default=0.0, help="noise on dataset")
    parser.add_argument("--noise_mult", type=float, default=1, help="Multiply size of test dataset by this, thanks to noise.")
    parser.add_argument("--num_goals", type=int, default=1, help="Number of goals in dataset")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--output_folder", type=str, default="", help="Where outputs will be saved")

    args = parser.parse_args()

    args.train_splits = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    args.output_folder = os.path.join(args.output_folder, f"{args.model}_{args.maze}_{args.num_seeds}s_{args.num_goals}g{args.add_goal}_{args.noise_std}n")  # where to save the plots
    args.data_folder = f"./data/data_{args.maze}/"  #  where to get the ground-truth policy

    if args.max_iter is None:
        args.max_iter = np.inf

    np.random.seed(args.meta_random_state)
    RANDOM_STATES = np.random.randint(0, high=1000000, size=args.num_seeds, dtype=int).tolist()
    data_plots = {}
    last_plots = {}

    # Ensure the plots directory exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # load data
    X_policy, y_policy, X_idm, y_idm = load_img_data(args.data_folder, add_goal=args.add_goal, num_goals=args.num_goals, noise_std=args.noise_std, noise_mult=args.noise_mult)
    n_samples_total = X_policy.shape[0]

    # plot images
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    axes = axes.flatten()
    for idx in range(18):
        current_state = X_idm[idx, :3]
        next_state = X_idm[idx, 3:]
        
        axes[2*idx].imshow(np.transpose(current_state, (1, 2, 0)))
        axes[2*idx].set_title(f'Current State')
        axes[2*idx + 1].imshow(np.transpose(next_state, (1, 2, 0)))
        axes[2*idx + 1].set_title(f'Next State')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{args.output_folder}/state_transitions_grid.png', dpi=100, bbox_inches='tight')
    plt.close()

    if args.model in ["LOGISTIC", "MLP"]:
        X_policy = X_policy.reshape(n_samples_total, -1)
        X_idm = X_policy.reshape(n_samples_total, -1)
    print(f"Test size: {len(y_policy)} (full dataset)")
    print(f"num features: {math.prod(X_policy.shape[1:])} (policy)")

    # Visualization of the ground-truth
    plot_labels(X_policy, y_policy, "Ground Truth Policy", args, filename=f"{args.output_folder}/ground_truth_policy.png")

    maze_size = (int(args.maze.split("x")[0]), int(args.maze.split("x")[1]))

    for TRAIN_SPLIT in args.train_splits:
        data_plots[TRAIN_SPLIT] = {}
        last_plots[TRAIN_SPLIT] = {}
        last_plots[TRAIN_SPLIT]["idm"] = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[])
        last_plots[TRAIN_SPLIT]["policy"] = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[])
        if not os.path.exists(f"{args.output_folder}/split_{TRAIN_SPLIT}"):
            os.makedirs(f"{args.output_folder}/split_{TRAIN_SPLIT}")
    
        for RANDOM_STATE in RANDOM_STATES:
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)

            # Randomly sample training set (seeded)
            train_idx = np.random.choice(range(n_samples_total), size=int(n_samples_total * TRAIN_SPLIT), replace=False)
            X_policy_train, y_policy_train = X_policy[train_idx], y_policy[train_idx]
            X_idm_train, y_idm_train = X_idm[train_idx], y_idm[train_idx]
            print(f"Train size: {len(y_policy_train)}")

            X_policy_test = X_policy
            y_policy_test = y_policy

            X_idm_test = X_idm
            y_idm_test = y_idm

            # Modeling
            if args.model == "LOGISTIC":
                policy_model = TorchLogistic(X_policy.shape[1], 4)
                idm_model = TorchLogistic(X_idm.shape[1], 4)
            elif args.model == "MLP":
                model_args = dict(hidden_dims=(300, 300, 300))
                policy_model = TorchMLP(X_policy.shape[1], 4, **model_args)
                idm_model = TorchMLP(X_idm.shape[1], 4, **model_args)
            elif args.model == "LINEAR_CNN":
                policy_model = TorchLogisticCNN(1, 4)
                idm_model = TorchLogisticCNN(2, 4)
            elif args.model == "CNN":
                policy_model = TorchCNN(1, 4, maze_size, num_conv_layers=3, conv_hidden_dim=128, num_fc_layers=2, fc_hidden_dim=128)
                idm_model = TorchCNN(2, 4, maze_size, num_conv_layers=3, conv_hidden_dim=128, num_fc_layers=2, fc_hidden_dim=128)
            else:
                raise ValueError("Unknown MODE: {}".format(args.model))

            print("Training policy...")
            policy_model, data_plots[TRAIN_SPLIT]["policy"] = train_torch_model(policy_model, X_policy_train, y_policy_train, X_policy_test, y_policy_test, batch_size=32, lr=1e-4, max_iter=args.max_iter)
            print("Training idm...")
            idm_model, data_plots[TRAIN_SPLIT]["idm"] = train_torch_model(idm_model, X_idm_train, y_idm_train, X_idm_test, y_idm_test, batch_size=32, lr=1e-4, max_iter=args.max_iter)
                
            # record last performance of the curve
            for key in last_plots[TRAIN_SPLIT]["policy"].keys():
                last_plots[TRAIN_SPLIT]["policy"][key].append(data_plots[TRAIN_SPLIT]["policy"][key]["curve"][-1])
            for key in last_plots[TRAIN_SPLIT]["idm"].keys():
                last_plots[TRAIN_SPLIT]["idm"][key].append(data_plots[TRAIN_SPLIT]["idm"][key]["curve"][-1])

        # Plot for only one seed (the last)
        # Plot loss and acc curves
        plot_loss_curves(data_plots, TRAIN_SPLIT, args)
        plot_acc_curves(data_plots, TRAIN_SPLIT, args)

        # Plot decision boundaries for the policy
        #plot_policy_decision_boundaries(X_policy_test, torch_predict(policy_model, X_policy_test), policy_model, args, X_train=X_policy_train, filename=f"{args.output_folder}/split_{TRAIN_SPLIT}/{args.model}_policy_predictions_{TRAIN_SPLIT}.png")

        # Plot idm predictions on the data
        #plot_labels(X_idm_test, torch_predict(idm_model, X_idm_test), f"{args.model} IDM Predictions", args, X_train=X_policy_train, filename=f"{args.output_folder}/split_{TRAIN_SPLIT}/{args.model}_idm_predictions_{TRAIN_SPLIT}.png")
    
    aggregated_plot(last_plots, args)

    # save data
    with open(f"{args.output_folder}/data_plots.pkl", "wb") as f:
        pickle.dump(data_plots, f)
    with open(f"{args.output_folder}/last_plots.pkl", "wb") as f:
        pickle.dump(last_plots, f)
    with open(f"{args.output_folder}/args.pkl", "wb") as f:
        pickle.dump(args, f)
    
    # load data
    # with open(f"{args.output_folder}/data_plots.pkl", "rb") as f:
    #     data_plots = pickle.load(f)
    # with open(f"{args.output_folder}/last_plots.pkl", "rb") as f:
    #     last_plots = pickle.load(f)
    

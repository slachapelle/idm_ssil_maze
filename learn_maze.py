import os
import pickle

import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
import argparse

from plot import aggregated_plot, plot_acc_curves, plot_loss_curves, plot_labels, plot_policy_decision_boundaries
from utils import torch_predict, load_data
from training import train_torch_model
from models import TorchLogistic, TorchMLP



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn maze policy with PyTorch.")
    parser.add_argument("--model", type=str, choices=["LOGISTIC", "MLP"], default="MLP", help="Model type: LOGISTIC or MLP")
    parser.add_argument("--max_iter", type=float, default=100000, help="max number of iterations. default is infinite.")
    parser.add_argument("--add_goal", action="store_true", default=False, help="Whether to add goal information (default: False)")
    parser.add_argument("--meta_random_state", type=int, default=9988338, help="Meta random seed")
    parser.add_argument("--num_inter", type=int, default=1, help="Number of interpolations per action")
    parser.add_argument("--maze", type=str, default="20x20", help="Maze name")
    parser.add_argument("--num_goals", type=int, default=1, help="Number of goals in dataset")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--output_folder", type=str, default="", help="Where outputs will be saved")

    args = parser.parse_args()

    args.train_splits = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    args.output_folder = os.path.join(args.output_folder, f"{args.model}_{args.maze}_{args.num_inter}x_{args.num_seeds}s_{args.num_goals}g{args.add_goal}")  # where to save the plots
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
    X_policy, y_policy, X_idm, y_idm = load_data(args.data_folder, num_inter=args.num_inter, add_goal=args.add_goal, num_goals=args.num_goals)
    n_samples_total = X_policy.shape[0]
    print(f"Test size: {len(y_policy)} (full dataset)")

    # Visualization of the ground-truth
    plot_labels(X_policy, y_policy, "Ground Truth Policy", args, filename=f"{args.output_folder}/ground_truth_policy.png")

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

            # Feature scaling
            scaler_policy = StandardScaler()
            scaler_idm = StandardScaler()

            X_policy_train = scaler_policy.fit_transform(X_policy_train)
            X_policy_test = scaler_policy.transform(X_policy)
            y_policy_test = y_policy

            X_idm_train = scaler_idm.fit_transform(X_idm_train)
            X_idm_test = scaler_idm.transform(X_idm)
            y_idm_test = y_idm

            # Modeling
            if args.model == "LOGISTIC":
                policy_model = TorchLogistic(X_policy.shape[1], 4)
                idm_model = TorchLogistic(X_idm.shape[1], 4)
            elif args.model == "MLP":
                model_args = dict(hidden_dims=(100, 100, 100, 100, 100))
                policy_model = TorchMLP(X_policy.shape[1], 4, **model_args)
                idm_model = TorchMLP(X_idm.shape[1], 4, **model_args)
            else:
                raise ValueError("Unknown MODE: {}".format(args.model))

            print("Training policy...")
            policy_model, data_plots[TRAIN_SPLIT]["policy"] = train_torch_model(policy_model, X_policy_train, y_policy_train, X_policy_test, y_policy_test, batch_size="full", lr=1e-3, max_iter=args.max_iter)
            print("Training idm...")
            idm_model, data_plots[TRAIN_SPLIT]["idm"] = train_torch_model(idm_model, X_idm_train, y_idm_train, X_idm_test, y_idm_test, batch_size="full", lr=1e-3, max_iter=args.max_iter)
            
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
        plot_policy_decision_boundaries(X_policy_test, torch_predict(policy_model, X_policy_test), policy_model, args, X_train=X_policy_train, filename=f"{args.output_folder}/split_{TRAIN_SPLIT}/{args.model}_policy_predictions_{TRAIN_SPLIT}.png")

        # Plot idm predictions on the data
        plot_labels(X_idm_test, torch_predict(idm_model, X_idm_test), f"{args.model} IDM Predictions", args, X_train=X_policy_train, filename=f"{args.output_folder}/split_{TRAIN_SPLIT}/{args.model}_idm_predictions_{TRAIN_SPLIT}.png")
    
    aggregated_plot(last_plots, args)

    # save data
    with open(f"{args.output_folder}/data_plots.pkl", "wb") as f:
        pickle.dump(data_plots, f)
    with open(f"{args.output_folder}/last_plots.pkl", "wb") as f:
        pickle.dump(last_plots, f)
    with open(f"{args.output_folder}/args.pkl", "wb") as f:
        pickle.dump(args, f)
    

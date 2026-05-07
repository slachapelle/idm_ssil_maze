import enum
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils import torch_predict, Direction, markers, colors, load_data

# Visualizations of the last model
def plot_policy_decision_boundaries(X, y, clf, args, resolution=0.02, filename="decision_boundaries.png", X_train=None):
    if args.add_goal:
        print("Not plotting decsision boundary when conditioning on goal (--add_goal).")
        return None
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = torch_predict(clf, np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=Direction(cl).name)
    # identify training samples
    if X_train is not None:
        plt.scatter(X_train[:, 0], X_train[:, 1], c='black', marker='x', s=20, label='Train samples')
    
    plt.legend()
    plt.title(f"{args.model} Policy Predictions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")

def plot_labels(X, y, title, args, filename="predictions.png", X_train=None):

    plt.figure(figsize=(8, 6))
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=Direction(cl).name)
    # identify training samples
    if X_train is not None:
        plt.scatter(X_train[:, 0], X_train[:, 1], c='black', marker='x', s=20, label='Train samples')

    plt.legend()
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")


def plot_loss_curves(data, split, args):
    policy_train = data[split]["policy"]["train_loss"]
    policy_test = data[split]["policy"]["test_loss"]
    idm_train = data[split]["idm"]["train_loss"]
    idm_test = data[split]["idm"]["test_loss"]
    postfix = f"loss_{split}"
    plt.figure(figsize=(8, 5))
    plt.plot(policy_train["iter"], policy_train["curve"], label=f"policy_train_{postfix}",c="blue")
    plt.plot(policy_test["iter"], policy_test["curve"], label=f"policy_test_{postfix}", c="blue", linestyle="dashed")
    plt.plot(idm_train["iter"], idm_train["curve"], label=f"idm_train_{postfix}", c="orange")
    plt.plot(idm_test["iter"], idm_test["curve"], label=f"idm_test_{postfix}", c="orange", linestyle="dashed")
    plt.xlabel("iter")
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.savefig(f"{args.output_folder}/split_{split}/{args.model}_train_test_{postfix}.png", dpi=300)


def plot_acc_curves(data, split, args):
    policy_train = data[split]["policy"]["train_acc"]
    policy_test = data[split]["policy"]["test_acc"]
    idm_train = data[split]["idm"]["train_acc"]
    idm_test = data[split]["idm"]["test_acc"]
    postfix = f"acc_{split}"
    plt.figure(figsize=(8, 5))
    plt.plot(policy_train["iter"], policy_train["curve"], label=f"policy_train_{postfix}",c="blue")
    plt.plot(policy_test["iter"], policy_test["curve"], label=f"policy_test_{postfix}", c="blue", linestyle="dashed")
    plt.plot(idm_train["iter"], idm_train["curve"], label=f"idm_train_{postfix}", c="orange")
    plt.plot(idm_test["iter"], idm_test["curve"], label=f"idm_test_{postfix}", c="orange", linestyle="dashed")
    plt.xlabel("iter")
    plt.legend()
    plt.grid()
    plt.savefig(f"{args.output_folder}/split_{split}/{args.model}_train_test_{postfix}.png", dpi=300)


def aggregated_plot(last_plots, args):
        # Plot last performance averaged over 
        plt.figure(figsize=(10, 6))
        plt.errorbar(args.train_splits, 
                    [np.mean(last_plots[s]["policy"]["test_acc"]) for s in args.train_splits],
                    [np.std(last_plots[s]["policy"]["test_acc"]) for s in args.train_splits], label="Policy Test Accuracy", capsize=3, marker=".")
        #plt.plot(args.train_splits, [np.mean(data_plots[s]["idm_test_acc"]) for s in args.train_splits], label="IDM Test Accuracy")
        plt.errorbar(args.train_splits, 
                    [np.mean(last_plots[s]["idm"]["test_acc"]) for s in args.train_splits],
                    [np.std(last_plots[s]["idm"]["test_acc"]) for s in args.train_splits], label="IDM Test Accuracy", capsize=3, marker=".")
        plt.title(f"{args.model} Test Accuracy vs Training Size")
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(f"{args.output_folder}/{args.model}_train_test_accuracy.png", dpi=300)
        print(f"Plot saved as {args.model}_train_test_accuracy.png")

def avg_rew_barplot(last_plots, args):
        # Plot last performance averaged over 
        plt.figure(figsize=(10, 6))
        plt.bar(['BC policy', 'IDM-based policy'], 
                    [np.mean(last_plots[m]["avg_rew"]) for m in ['policy', 'idm_policy']],
                    yerr=[np.std(last_plots[m]["avg_rew"]) for m in ['policy', 'idm_policy']])
        #plt.title(f"{args.model}")
        #plt.xlabel("Training Size")
        plt.ylabel("Average reward")
        #plt.legend()
        #plt.grid()
        plt.savefig(f"{args.output_folder}/{args.model}_avg_rew.png", dpi=300)
        print(f"Plot saved as {args.model}_avg_rew.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn maze policy with PyTorch.")
    parser.add_argument("--output_folder", type=str, default="", help="path")
    args = parser.parse_args()

    # load output of experiment
    with open(f"{args.output_folder}/args.pkl", "rb") as f:
        args = pickle.load(f)
    with open(f"{args.output_folder}/data_plots.pkl", "rb") as f:
        data_plots = pickle.load(f)
    with open(f"{args.output_folder}/last_plots.pkl", "rb") as f:
        last_plots = pickle.load(f)

    # load data (Only to plot ground-truth)
    X_policy, y_policy, X_idm, y_idm = load_data(args.data_folder, num_inter=args.num_inter)
    n_samples_total = X_policy.shape[0]

    # plots
    plot_labels(X_policy, y_policy, "Ground Truth Policy", args, X_train=None, filename=f"{args.output_folder}/ground_truth_policy.png")
    aggregated_plot(last_plots, args)
    for TRAIN_SPLIT in args.train_splits:
        plot_loss_curves(data_plots, TRAIN_SPLIT, args)
        plot_acc_curves(data_plots, TRAIN_SPLIT, args)



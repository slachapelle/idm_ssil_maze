import enum
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils import torch_predict, Direction, markers, colors, load_data, load_maze_img
from plot import plot_labels, aggregated_plot, plot_loss_curves, plot_acc_curves
from learn_maze_stoch_expert import expert_policy_stoch, generate_dataset, GridEnv

# Increase matplotlib font sizes for readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
})

ELINEWIDTH = None
CAPTHICK = None

def aggregated_plot_final(last_plots, args, ax, title="", metric="test_acc", postfix="", linestyle="solid", more_data=False):
        # Plot last performance averaged over 
        idm_means = [np.mean(last_plots[s]["idm"][metric]) for s in args.train_splits]
        idm_stds = [np.std(last_plots[s]["idm"][metric]) for s in args.train_splits]
        policy_means = [np.mean(last_plots[s]["policy"][metric]) for s in args.train_splits]
        policy_stds = [np.std(last_plots[s]["policy"][metric]) for s in args.train_splits]
        container_idm = ax.errorbar(args.train_splits, idm_means, idm_stds, 
                    label=rf'VM$\!^*\!$-IDM{postfix}', capsize=3, marker=".", color="tab:orange", linestyle=linestyle, elinewidth=ELINEWIDTH, capthick=CAPTHICK)
        container_policy = ax.errorbar(args.train_splits, policy_means, policy_stds, 
                    label="BC"+postfix, capsize=3, marker=".", color="tab:blue", linestyle=linestyle, elinewidth=ELINEWIDTH, capthick=CAPTHICK)
        #plt.plot(args.train_splits, [np.mean(data_plots[s]["idm_test_acc"]) for s in args.train_splits], label="IDM Test Accuracy")
        ax.set_title(title)
        if metric == "test_acc":
            ax.set_ylim(0.2, 1.05)
        elif metric == "test_loss":
            pass
            #ax.set_ylim(-1, max([max(idm_means), max(policy_means)]))
            #ax.set_ylim(-1, 20)
            #ax.set_yscale("log")
        #ax.set_xlabel("Training Size")
        #ax.set_ylabel("Test accuracy")
        if more_data:
            ax.set_xscale("log", base=2)
            ax.set_xticks([2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**0])
        else:
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # errorbars transparent
        for i in container_idm[1] + container_idm[2] + container_policy[1] + container_policy[2]:
            i.set_alpha(0.5)
        return ax

def aggregated_plot_goal(last_plots, args, ax, title="", metric="test_acc", postfix="", linestyle="solid"):
        if args.add_goal:
            bc_label = rf'BC$_G${postfix}'
            vmidm_label = rf'VM$_G^*$-IDM$_G${postfix}'
        else:
            bc_label = rf'BC{postfix}'
            vmidm_label = rf'VM$_G^*$-IDM{postfix}'

        # Plot last performance averaged over
        idm_means = [np.mean(last_plots[s]["idm"][metric]) for s in args.train_splits]
        idm_stds = [np.std(last_plots[s]["idm"][metric]) for s in args.train_splits]
        policy_means = [np.mean(last_plots[s]["policy"][metric]) for s in args.train_splits]
        policy_stds = [np.std(last_plots[s]["policy"][metric]) for s in args.train_splits]
        container_idm = ax.errorbar(args.train_splits, idm_means, idm_stds, label=vmidm_label, capsize=3, marker=".", color="tab:orange", linestyle=linestyle, elinewidth=ELINEWIDTH, capthick=CAPTHICK)
        container_policy = ax.errorbar(args.train_splits, policy_means, policy_stds, label=bc_label, capsize=3, marker=".", color="tab:blue", linestyle=linestyle, elinewidth=ELINEWIDTH, capthick=CAPTHICK)
        ax.set_title(title)
        if metric == "test_acc":
            ax.set_ylim(0.2, 1.05)
        elif metric == "test_loss":
            ax.set_ylim(-1, 10)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # errorbars transparent
        for i in container_idm[1] + container_idm[2] + container_policy[1] + container_policy[2]:
            i.set_alpha(0.5)
        return ax

def aggregated_plot_stoch(last_plots, args, ax, title="", metric="test_acc", postfix="", linestyle="solid"):
        # Plot last performance averaged over 
        idm_means = [np.mean(last_plots[s]["idm_policy"][metric]) for s in args.train_splits]
        idm_stds = [np.std(last_plots[s]["idm_policy"][metric]) for s in args.train_splits]
        policy_means = [np.mean(last_plots[s]["policy"][metric]) for s in args.train_splits]
        policy_stds = [np.std(last_plots[s]["policy"][metric]) for s in args.train_splits]
        container_idm = ax.errorbar(args.train_splits, idm_means, idm_stds, label="IDM Lab."+postfix, capsize=3, marker=".", color="tab:orange", linestyle=linestyle, elinewidth=ELINEWIDTH, capthick=CAPTHICK)
        container_policy = ax.errorbar(args.train_splits, policy_means,  policy_stds, label="BC"+postfix, capsize=3, marker=".", color="tab:blue", linestyle=linestyle, elinewidth=ELINEWIDTH, capthick=CAPTHICK)
        #plt.plot(args.train_splits, [np.mean(data_plots[s]["idm_test_acc"]) for s in args.train_splits], label="IDM Test Accuracy")
        
        ax.set_title(title)
        if metric == 'avg_rew':
            ax.set_ylim(-0.05, 1.05)
        if metric == 'avg_dist':
            ax.set_ylim(-0.5, 10)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # errorbars transparent
        for i in container_idm[1] + container_idm[2] + container_policy[1] + container_policy[2]:
            i.set_alpha(0.5)
        return ax


def plot_count_heatmap(ax, X_train, env_size, prob=0.5, annotate=False):
            counts = np.zeros((env_size, env_size), dtype=int)
            # X_train rows are (row, col)
            for r, c in X_train:
                counts[int(r), int(c)] += 1
            total_count = np.sum(counts)
            freqs = counts/total_count
            im = ax.imshow(freqs, cmap="viridis", origin="upper")
            ax.set_ylabel(rf'$p(\mathrm{{right}}) = {prob}$')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            return ax


if __name__ == "__main__":
    ### Maze complexity plot, no conditioning ###
    mazes = ["10x10", "20x20", "50x50"]
    exp_root = 'exp/results_states'
    metric = "test_acc"
    if metric == "test_acc":
        y_label = "Test acc."
    elif metric == "test_loss":
        y_label = "Test KL"
    else:
        y_label = "????????"

    # create maze images
    fig, axes = plt.subplots(1, 3, figsize=(10, 2), dpi=300)
    for i, maze in enumerate(mazes):
        # get data folder
        output_folder = f"{exp_root}/MLP_{maze}_1x_5s_1gFalse"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            exp_args = pickle.load(f)
        data_folder = exp_args.data_folder

        # load maze image
        maze_img = load_maze_img(data_folder)
        maze_img = maze_img[:, :, [2, 1, 0]]  # fix RGB
        print(maze_img.min(), maze_img.max(), maze_img.shape)

        # plot
        axes[i].imshow(maze_img)
        #axes[0, i].set_title(f"{maze} maze")
        axes[i].set_ylabel(f"{maze}")
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])
    
    fig.tight_layout()
    fig.savefig(f"{exp_root}/maze_imgs")
    fig.clf()

    # create plot showing maze image next to test accuracy curves for different data split.
    args = {"mlp":{}, "logistic": {}}
    last_plots = {"mlp":{}, "logistic": {}}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.3), dpi=300)

    for i, maze in enumerate(mazes):
        # load output of experiment
        output_folder = f"{exp_root}/MLP_{maze}_1x_5s_1gFalse"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            args['mlp'][maze] = pickle.load(f)
        with open(f"{output_folder}/last_plots.pkl", "rb") as f:
            last_plots['mlp'][maze] = pickle.load(f)
        
        output_folder = f"{exp_root}/LOGISTIC_{maze}_1x_5s_1gFalse"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            args['logistic'][maze] = pickle.load(f)
        with open(f"{output_folder}/last_plots.pkl", "rb") as f:
            last_plots['logistic'][maze] = pickle.load(f)
        
        axes[i] = aggregated_plot_final(last_plots['logistic'][maze], args["logistic"][maze], axes[i], metric=metric, postfix=" (LC)")
        axes[i] = aggregated_plot_final(last_plots['mlp'][maze], args["mlp"][maze], axes[i], metric=metric, postfix=" (5L MLP)", linestyle='dashed')
        if i == 0:
            axes[i].set_ylabel(y_label)
        axes[i].set_xlabel("Train split")
        axes[i].grid()
    fig.tight_layout()
    # Reorder by index
    handles, labels = axes[1].get_legend_handles_labels()
    order = [1, 3, 0, 2]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.4)
    fig.savefig(f"{exp_root}/maze_size_plot_{metric}")
    fig.clf()
    print(f"saved in {exp_root}/maze_size_plot_{metric}")
    
    # image based exp
    exp_root = 'exp/results_images'
    args = {"cnn":{}, "linear_max_cnn": {}}
    last_plots = {"cnn":{}, "linear_max_cnn": {}}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.3), dpi=300)

    for i, maze in enumerate(mazes):
        # load output of experiment
        output_folder = f"{exp_root}/CNN_{maze}_5s_1gFalse_0.0n"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            args['cnn'][maze] = pickle.load(f)
        with open(f"{output_folder}/last_plots.pkl", "rb") as f:
            last_plots['cnn'][maze] = pickle.load(f)
        
        output_folder = f"{exp_root}/LINEAR_CNN_{maze}_5s_1gFalse_0.0n"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            args['linear_max_cnn'][maze] = pickle.load(f)
        with open(f"{output_folder}/last_plots.pkl", "rb") as f:
            last_plots['linear_max_cnn'][maze] = pickle.load(f)

        axes[i] = aggregated_plot_final(last_plots['linear_max_cnn'][maze], args['linear_max_cnn'][maze], axes[i], metric=metric, postfix=" (1L CNN)")
        axes[i] = aggregated_plot_final(last_plots['cnn'][maze], args["cnn"][maze], axes[i], metric=metric, postfix=" (5L CNN)", linestyle='dashed')
        if i == 0:
            axes[i].set_ylabel(y_label)
        axes[i].set_xlabel("Train split")
        axes[i].grid()
    fig.tight_layout()
    # reordering labels
    handles, labels = axes[1].get_legend_handles_labels()
    order = [1, 3, 0, 2]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.4)
    fig.savefig(f"{exp_root}/maze_size_plot_img_{metric}")
    fig.clf()
    print(f"saved in {exp_root}/maze_size_plot_img_{metric}")


    ### Plot goal conditioning ###
    goals = [1, 3, 10]
    args = {}
    last_plots = {}
    exp_root = 'exp/results_goal'

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), dpi=300)
    for i, goal in enumerate(goals):
        for goal_conditioned in [False, True]:
            # load output of experiment
            output_folder = f"{exp_root}/MLP_10x10_1x_5s_{goal}g{goal_conditioned}"
            with open(f"{output_folder}/args.pkl", "rb") as f:
                args[goal] = pickle.load(f)
            with open(f"{output_folder}/last_plots.pkl", "rb") as f:
                last_plots[goal] = pickle.load(f)
            
            if goal_conditioned:
                linestyle = "dashed"
            else:
                linestyle = "solid"
            
            axes[i] = aggregated_plot_goal(last_plots[goal], args[goal], axes[i], metric=metric, linestyle=linestyle, postfix=" (5L MLP)")
            if i == 0:
                axes[i].set_ylabel(y_label)
            axes[i].set_xlabel("Train split")
        axes[i].set_title(f"Number of goals = {goal}")
        axes[i].grid()
    fig.tight_layout()
    # reordering labels
    handles, labels = axes[1].get_legend_handles_labels()
    order = [1, 3, 0, 2]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.37)
    fig.savefig(f"{exp_root}/goal_plot")
    print(f"saved in {exp_root}/goal_plot")

    
    ### Plot for stochastic environment V2 ###
    # create plot showing maze image next to test accuracy curves for different data split.
    exp_root = 'exp/results_stoch_env'
    probs = [0.25, 0.5, 1.0]
    maze = "50x50"
    y_label = "Test acc."
    #metric = 'test_acc'
    metric = 'test_acc2'
    args = {"mlp":{}, "logistic": {}}
    last_plots = {"mlp":{}, "logistic": {}}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), dpi=300)

    for i, prob in enumerate(probs):
        # load output of experiment
        output_folder = f"{exp_root}/MLP_{maze}_{prob}p_4n_5seeds"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            args['mlp'][prob] = pickle.load(f)
        with open(f"{output_folder}/last_plots.pkl", "rb") as f:
            last_plots['mlp'][prob] = pickle.load(f)
        
        output_folder = f"{exp_root}/LOGISTIC_{maze}_{prob}p_4n_5seeds"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            args['logistic'][prob] = pickle.load(f)
        with open(f"{output_folder}/last_plots.pkl", "rb") as f:
            last_plots['logistic'][prob] = pickle.load(f)
        
        axes[i] = aggregated_plot_final(last_plots['logistic'][prob], args["logistic"][prob], axes[i], metric=metric, postfix=" (LC)", more_data=True)
        axes[i] = aggregated_plot_final(last_plots['mlp'][prob], args["mlp"][prob], axes[i], metric=metric, postfix=" (5L MLP)", linestyle='dashed', more_data=True)
        if i == 0:
            axes[i].set_ylabel(y_label)
        axes[i].set_xlabel("Train split")
        axes[i].set_title(f"p(no ops) = {prob}")
        axes[i].grid()
    fig.tight_layout()
    # reordering labels
    handles, labels = axes[1].get_legend_handles_labels()
    order = [1, 3, 0, 2]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.37)
    fig.savefig(f"{exp_root}/maze_size_plot_{metric}")
    fig.clf()
    print(f"saved in {exp_root}/maze_size_plot_{metric}")


    ### Plot stochastic expert experiment ###
    exp_root = 'exp/results_stoch_expert'
    n_seeds = 10
    probs = [0.5, 0.9, 1.0]
    args = {}
    last_plots = {}
    y_label = "Avg reward"
    metric = 'avg_rew'
    #metric = 'avg_dist'
    env = GridEnv(20)

    # path distribution
    fig, axes = plt.subplots(1, 3, figsize=(10, 2), dpi=300)
    for i, prob in enumerate(probs):
        
        expert_policy = lambda state, env_size: expert_policy_stoch(state, env_size, prob) 
        X, _, _ = generate_dataset(env=env, policy=expert_policy, n_trajectories=1000, seed=198549382)
        axes[i] = plot_count_heatmap(axes[i], X, 20, prob=prob)
    fig.tight_layout()
    fig.savefig(f"{exp_root}/sochastic_plot_img")
    fig.clf()
    print(f"saved in {exp_root}/sochastic_plot_img")


    fig, axes = plt.subplots(1, 3, figsize=(10, 2.3), dpi=300)
    for i, prob in enumerate(probs):
        # load output of experiment
        output_folder = f"{exp_root}/MLP_20env_{prob}p_1000n_{n_seeds}seeds"
        with open(f"{output_folder}/args.pkl", "rb") as f:
            args[prob] = pickle.load(f)
        with open(f"{output_folder}/last_plots.pkl", "rb") as f:
            last_plots[prob] = pickle.load(f)
        
        axes[i] = aggregated_plot_stoch(last_plots[prob], args[prob], axes[i], metric=metric, postfix=" (5L MLP)")
        if i == 0:
            axes[i].set_ylabel(y_label)
        axes[i].set_xlabel("Train split")
        axes[i].grid()
    fig.tight_layout()
    # reordering labels
    handles, labels = axes[1].get_legend_handles_labels()
    order = [1, 0]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.4)
    fig.savefig(f"{exp_root}/expert_stoch_plot_{metric}")
    fig.clf()
    print(f"saved in {exp_root}/expert_stoch_plot_{metric}")
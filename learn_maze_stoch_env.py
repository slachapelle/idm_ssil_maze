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

from training import train_torch_model, test_loss_eval, test_accuracy_vm_idm_stoch_env_v2, test_accuracy_vm_idm_stoch_env_v2_sample_VM, test_accuracy_policy
from utils import torch_predict, load_data_stoch_env, load_data
from plot import avg_rew_barplot
from models import TorchMLP, TorchLogistic


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn maze policy with PyTorch.")
    parser.add_argument("--model", type=str, choices=["LOGISTIC", "MLP"], default="MLP", help="Model type: LOGISTIC or MLP")
    parser.add_argument("--max_iter", type=float, default=20000, help="max number of iterations. default is infinite.")
    parser.add_argument("--meta_random_state", type=int, default=9988338, help="Meta random seed")
    parser.add_argument("--maze", type=str, default="20x20", help="Maze name")
    parser.add_argument("--env_prob", type=float, default=0.5, help="stochasticity of the env")
    parser.add_argument("--env_version", type=str, default="v1", help="type of env stochasticity")
    #parser.add_argument("--num_samples_test", type=int, default=10*100**2, help="Num samples total")
    parser.add_argument("--samples_mult", type=int, default=1)
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--output_folder", type=str, default="", help="Where outputs will be saved")

    args = parser.parse_args()
    args.output_folder = os.path.join(args.output_folder, f"{args.model}_{args.maze}_{args.env_prob}p_{args.samples_mult}n_{args.num_seeds}seeds")  # where to save the plots
    args.data_folder = f"./data/data_{args.maze}/"  #  where to get the ground-truth policy

    args.train_splits = [2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**0]

    np.random.seed(args.meta_random_state)
    RANDOM_STATES = np.random.randint(0, high=1000000, size=args.num_seeds, dtype=int).tolist()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    data_plots = {}
    last_plots = {}

    for TRAIN_SPLIT in args.train_splits:
        data_plots[TRAIN_SPLIT] = {}
        last_plots[TRAIN_SPLIT] = {}
        last_plots[TRAIN_SPLIT]["policy"] = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[], kl=[], test_acc2=[])
        last_plots[TRAIN_SPLIT]["idm"] = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[], kl=[], test_acc2=[], test_acc3=[])
        if not os.path.exists(f"{args.output_folder}/split_{TRAIN_SPLIT}"):
            os.makedirs(f"{args.output_folder}/split_{TRAIN_SPLIT}")
    
        for RANDOM_STATE in RANDOM_STATES:
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)

            seed_folder = f"{args.output_folder}/split_{TRAIN_SPLIT}/seed{RANDOM_STATE}"
            if not os.path.exists(seed_folder):
                os.makedirs(seed_folder)

            # Randomly sample data (seeded)
            X_policy, y_policy, X_idm, y_idm = load_data_stoch_env(args.data_folder, prob=args.env_prob, samples_mult=args.samples_mult, env_version=args.env_version)
            total_trainset_size = X_policy.shape[0]
            X_policy_test, y_policy_test, X_idm_test, y_idm_test = load_data_stoch_env(args.data_folder, prob=args.env_prob, samples_mult=10 * args.samples_mult, env_version=args.env_version)
            total_testset_size = X_policy_test.shape[0]

            # Loading the dataset without stochasticity. (all states appears exactly once in this dataset)
            X_policy_ops, y_ops, X_idm_ops, _ = load_data(args.data_folder, num_inter=1, num_goals=1)
            # create dataset (s, s') where s = s' (no-ops dataset)
            X_idm_no_ops = np.concatenate([X_idm_ops[:, 0:2], X_idm_ops[:, 0:2]], axis=1)
            
            # shuffle trainset
            perm = np.random.choice(range(total_trainset_size), size=total_trainset_size, replace=False)
            X_policy = X_policy[perm]
            X_idm = X_idm[perm]
            y_policy = y_policy[perm]
            y_idm = y_idm[perm]  # note that y_idm == y_policy
            
            # select labeled samples
            labeled_trainset_size = int(total_trainset_size * TRAIN_SPLIT)
            X_policy_lab = X_policy[:labeled_trainset_size]
            X_idm_lab = X_idm[:labeled_trainset_size]
            y_policy = y_policy[:labeled_trainset_size]  # throwing out bunch of labels
            y_idm = y_idm[:labeled_trainset_size]  # throwing out bunch of labels

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

            #X_policy = policy_model.scaler.fit_transform(X_policy)
            X_policy_lab = policy_model.scaler.fit_transform(X_policy_lab)
            X_policy_test = policy_model.scaler.transform(X_policy_test)
            X_policy_ops = policy_model.scaler.transform(X_policy_ops)

            #X_idm = idm_model.scaler.fit_transform(X_idm)
            X_idm_lab = idm_model.scaler.fit_transform(X_idm_lab)
            X_idm_test = idm_model.scaler.transform(X_idm_test)
            X_idm_ops = idm_model.scaler.transform(X_idm_ops)
            X_idm_no_ops = idm_model.scaler.transform(X_idm_no_ops)

            print("BC...")
            _t0 = time.time()
            policy_model, data_plots[TRAIN_SPLIT]["policy"] = train_torch_model(policy_model, X_policy_lab, y_policy, X_val=X_policy_test, y_val=y_policy_test, max_iter=args.max_iter, batch_size=512, env=None)
            print(f"   done in {time.time() - _t0:.2f}s")

            print("Train IDM...")
            _t0 = time.time()
            idm_model, data_plots[TRAIN_SPLIT]["idm"] = train_torch_model(idm_model, X_idm_lab, y_idm, X_val=X_idm_test, y_val=y_idm_test, max_iter=args.max_iter, batch_size=512, env=None)
            print(f"   done in {time.time() - _t0:.2f}s")

            print("Final eval...")
            _t0 = time.time()
            # Estimate KL divergences on the fixed test set
            policy_ce = test_loss_eval(policy_model, X_policy_test, y_policy_test, batch_size=1024)
            idm_ce = test_loss_eval(idm_model, X_idm_test, y_idm_test, batch_size=1024)
            # KL(π*||π) = cross-entropy(π*, π) − H(π*) = CE - 0 (π* is deterministic)
            kl_policy = policy_ce
            # KL(h*||h) = cross-entropy − H(h*) = CE − 0  (h* is deterministic)
            kl_idm = idm_ce
            
            # the test accuracy of the IDM is not the same as the test accuracy of the VM*-IDM policy here.
            # This is because the the ground-truth video model v*(s' | s) is not deterministic
            # We design specialized functions to compute the test accuracy of the VM*-IDM. 
            if args.env_version == 'v1':
                vm_idm_test_acc = -1   # TODO:
            elif args.env_version == 'v2': 
                vm_idm_test_acc = test_accuracy_vm_idm_stoch_env_v2(idm_model, X_idm_ops, X_idm_no_ops, y_ops, args.env_prob)
                vm_idm_test_acc_sample_vm = test_accuracy_vm_idm_stoch_env_v2_sample_VM(idm_model, X_idm_ops, X_idm_no_ops, y_ops, args.env_prob)
            policy_test_acc = test_accuracy_policy(policy_model, X_policy_ops, y_ops)

            print(f"   done in {time.time() - _t0:.2f}s")

            # record last performance of the curve
            _derived = {"kl", "test_loss", 'avg_rew', 'avg_dist', 'test_acc2', 'test_acc3'}
            for key in last_plots[TRAIN_SPLIT]["policy"].keys():
                if key not in _derived:
                    last_plots[TRAIN_SPLIT]["policy"][key].append(data_plots[TRAIN_SPLIT]["policy"][key]["curve"][-1])
            for key in last_plots[TRAIN_SPLIT]["idm"].keys():
                if key not in _derived:
                    last_plots[TRAIN_SPLIT]["idm"][key].append(data_plots[TRAIN_SPLIT]['idm'][key]["curve"][-1])
            last_plots[TRAIN_SPLIT]["policy"]["kl"].append(kl_policy)
            last_plots[TRAIN_SPLIT]["idm"]["kl"].append(kl_idm)
            last_plots[TRAIN_SPLIT]["policy"]["test_loss"].append(policy_ce)
            last_plots[TRAIN_SPLIT]["idm"]["test_loss"].append(idm_ce)
            last_plots[TRAIN_SPLIT]["policy"]["test_acc2"].append(policy_test_acc)  # should be very similar to test_acc
            last_plots[TRAIN_SPLIT]["idm"]["test_acc2"].append(vm_idm_test_acc)
            last_plots[TRAIN_SPLIT]["idm"]["test_acc3"].append(vm_idm_test_acc_sample_vm)
            
            # last_plots[TRAIN_SPLIT]["idm"]["test_acc"] is the accuracy of the IDM
            # last_plots[TRAIN_SPLIT]["idm"]["test_acc2"] is the accuracy of the VM*-IDM, which is different since v*(s' | s) is stochastic.
            
            plot_learning_curves(data_plots[TRAIN_SPLIT]["policy"], seed_folder, model_name="policy")
            plot_learning_curves(data_plots[TRAIN_SPLIT]["idm"], seed_folder, model_name="idm")
        
            with open(f"{seed_folder}/data_plots.pkl", "wb") as f:
                pickle.dump(data_plots[TRAIN_SPLIT], f)
            torch.save(policy_model, f"{seed_folder}/policy_model.pt")
            torch.save(idm_model, f"{seed_folder}/idm_model.pt")
    
    # plot avg_reward
    print(last_plots)
    with open(f"{args.output_folder}/last_plots.pkl", "wb") as f:
        pickle.dump(last_plots, f)
    with open(f"{args.output_folder}/args.pkl", "wb") as f:
        pickle.dump(args, f)

    

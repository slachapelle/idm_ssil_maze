import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

from utils import torch_predict


def eval(model, criterion, device, X_train, y_train, X_val, y_val, train_acc_, val_loss_, val_acc_, avg_rew_, avg_dist_, iter, env=None):
    # Validation loss
    model.eval()

    with torch.no_grad():
        # _t0 = time.time()
        full_train_out = model(X_train.to(device))
        full_train_loss = criterion(full_train_out, y_train.to(device)).item()
        # print(f"   full_train_out done in {time.time() - _t0:.2f}s")
        #full_train_loss_["curve"].append(full_train_loss)
        #full_train_loss_["iter"].append(iter)

        if X_val is not None:
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, y_val.to(device)).item()
            val_loss_["curve"].append(val_loss)
            val_loss_["iter"].append(iter)

            val_acc = accuracy_score(y_val.cpu().numpy(), torch_predict(model, X_val))
            val_acc_["curve"].append(val_acc)
            val_acc_["iter"].append(iter)
        else:
            val_acc = None
        #import ipdb; ipdb.set_trace()
        # _t0 = time.time()
        train_acc = accuracy_score(y_train.cpu().numpy(), torch_predict(model, X_train))
        # print(f"   train_acc done in {time.time() - _t0:.2f}s")
        train_acc_["curve"].append(train_acc)
        train_acc_["iter"].append(iter)

        if env is not None:
            # _t0 = time.time()
            avg_rew, avg_dist = env.evaluate_policy(model, num_episodes=25, device=device)
            # print(f"   evaluate_policy done in {time.time() - _t0:.2f}s")

            avg_rew_["curve"].append(avg_rew)
            avg_rew_["iter"].append(iter)
            avg_dist_["curve"].append(avg_dist)
            avg_dist_["iter"].append(iter)

    return full_train_loss, train_acc, val_acc

def test_loss_eval(model, X, y, batch_size=512):
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        total_loss = 0.0
        num_samples = 0
        n = X.shape[0]
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            out = model(xb)
            loss = criterion(out, yb)
            num_samples += yb.shape[0]
            total_loss += loss.item()
        return total_loss / num_samples

def test_accuracy_vm_idm_stoch_env_v2(idm, X_idm_ops, X_idm_no_ops, y, prob):
    """
    Test accuracy of the vmidm policy when the vm is ground-truth, in args.env_version == 'v2'
    We first compute the probability given by the IDM-based policy, then predict taking argmax.
    """
    with torch.no_grad():
        idm.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_idm_ops = torch.tensor(X_idm_ops, dtype=torch.float32).to(device)
        X_idm_no_ops = torch.tensor(X_idm_no_ops, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        
        logit_ops = idm(X_idm_ops)
        logit_no_ops = idm(X_idm_no_ops)
        vm_idm_policy_proba = prob * torch.softmax(logit_no_ops, dim=1) + (1 - prob) * torch.softmax(logit_ops, dim=1)
        action_pred = torch.argmax(vm_idm_policy_proba, dim=1)
        acc = torch.mean((action_pred == y).float())
        return acc.item()

def test_accuracy_vm_idm_stoch_env_v2_sample_VM(idm, X_idm_ops, X_idm_no_ops, y, prob):
    """
    Alternative way to compute test accuracy of the vmidm policy when the vm is ground-truth, in args.env_version == 'v2'
    To make a prediction, we first sample s' from ground-truth vm, compute h_hat(a | s, s') and output the argmax over a.
    This prediction is clearly stochastic, so we report it's average accuracy.
    This is actually equivalent to the accuracy of the IDM. 
    """
    with torch.no_grad():
        idm.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_idm_ops = torch.tensor(X_idm_ops, dtype=torch.float32).to(device)
        X_idm_no_ops = torch.tensor(X_idm_no_ops, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        logit_ops = idm(X_idm_ops)
        action_pred_ops = torch.argmax(logit_ops, dim=1)
        acc_ops = torch.mean((action_pred_ops == y).float())

        logit_no_ops = idm(X_idm_no_ops)
        action_pred_no_ops = torch.argmax(logit_no_ops, dim=1)
        acc_no_ops = torch.mean((action_pred_no_ops == y).float())

        print("Prediction agreement between h(a|s,s) and h(a|s,s'):", torch.mean((action_pred_no_ops == action_pred_ops).float()).item())

        acc = prob * acc_no_ops + (1 - prob) * acc_ops
        return acc.item()

def test_accuracy_policy(policy, X_policy_ops, y):
    with torch.no_grad():
        policy.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_policy_ops = torch.tensor(X_policy_ops, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logit = policy(X_policy_ops)
        bc_policy_proba = torch.softmax(logit, dim=1)
        action_pred = torch.argmax(bc_policy_proba, dim=1)
        acc = torch.mean((action_pred == y).float())
        return acc.item()

def train_torch_model(model, X_train, y_train, X_val=None, y_val=None, batch_size="full", lr=1e-3, max_iter=np.inf, env=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size == "full":
        batch_size = X_train.shape[0]  # full batch
    val_freq = 500
    model = model.to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    if X_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    n_train = X_train.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loss_ = {"curve": [], "iter":[]}
    #full_train_loss_ = {"curve": [], "iter":[]}
    train_acc_ = {"curve": [], "iter":[]}
    val_loss_ = {"curve": [], "iter":[]}
    val_acc_ = {"curve": [], "iter":[]}
    avg_rew_ = {"curve": [], "iter": []}
    avg_dist_ = {"curve": [], "iter": []}
    train_acc = 0.0
    iter = 0
    epoch = 0
    # _step_time_acc = 0.0
    # _data_time_acc = 0.0
    
    t0 = time.time()
    while True:
        for i in range(0, n_train, batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            model.train()
            # _t0 = time.time()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            # _step_time_acc += time.time() - _t0
            # _td = time.time()

            train_loss_["curve"].append(loss.item())
            train_loss_["iter"].append(iter)


            if iter % val_freq == 0:
                print(f"    loop without eval = {time.time()- t0:.2f}s")
                t0 = time.time()
                full_train_loss, train_acc, val_acc = eval(model, criterion, device, X_train, y_train, X_val, y_val, train_acc_, val_loss_, val_acc_, avg_rew_, avg_dist_, iter, env=env)
                print(f"    eval time {time.time() - t0:.2f}s")
                t0 = time.time()
                print(f"iter: {iter}, loss: {loss.item()}, train acc: {train_acc}, test acc: {val_acc}, out: {out[np.random.randint(0, out.shape[0])]}")

                if iter >= max_iter or full_train_loss < 1e-7:
                    return model, {"train_loss": train_loss_, "train_acc": train_acc_, "test_loss": val_loss_, "test_acc": val_acc_, "avg_rew": avg_rew_, "avg_dist": avg_dist_}

            iter += 1
        epoch += 1

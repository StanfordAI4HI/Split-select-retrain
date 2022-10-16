import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from os.path import join as pjoin

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm, t

from pois.offline_pg import set_random_seed, MLPPolicy, wis_ope, clipped_is_ope, cwpdis_ope


class BCDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False):
        self.df = df
        self.model_round_as_feature = model_round_as_feature

        feature_names = ['pre', 'anxiety', 'thinking', 'last_step']

        feature_names = feature_names

        self.feature_names = feature_names

        if model_round_as_feature:
            self.feature_names += ['model_round']

        self.target_names = ['p_encourage', 'p_hint', 'p_guided_prompt']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        features = row[self.feature_names]
        targets = row[self.target_names]

        features = features.to_numpy().astype(float)
        targets = targets.to_numpy().astype(float)

        return {'features': torch.from_numpy(features).float(),
                'targets': torch.from_numpy(targets).float()}


feature_names = ['pre', 'anxiety', 'thinking', 'last_step']
target_names = ['p_encourage', 'p_hint', 'p_guided_prompt']

MAX_TIME = 20


def compute_is_weights_for_nn_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='reward', no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                     model_round_as_feature=False):
    df = behavior_df
    user_ids = df['user_id'].unique()
    n = len(user_ids)

    MAX_TIME = max(behavior_df.groupby('user_id').size())

    assert reward_column in ['reward']

    pies = torch.zeros((n, MAX_TIME))

    pibs = torch.zeros((n, MAX_TIME))
    rewards = torch.zeros((n, MAX_TIME))
    lengths = np.zeros((n))

    user_rewards = df.groupby("user_id")[reward_column].sum()
    train_reward_mu = user_rewards.mean()
    train_reward_std = user_rewards.std()

    for idx, user_id in enumerate(user_ids):
        data = df[df['user_id'] == user_id]

        if not model_round_as_feature:
            features = np.asarray(data[feature_names]).astype(float)
        else:
            raise Exception("No model round in this dataset")

        targets = np.asarray(data[target_names]).astype(float)
        actions = np.asarray(data['action']).astype(int)

        length = features.shape[0]
        lengths[idx] = length

        T = targets.shape[0]

        beh_probs = torch.from_numpy(np.array([targets[i, a] for i, a in enumerate(actions)])).float()
        pibs[idx, :T] = beh_probs

        gr_mask = None
        if gr_safety_thresh > 0:
            if not is_train:

                if not use_knn:
                    gr_mask = None
                else:
                    raise Exception("not implemented")










            else:
                beh_action_probs = torch.from_numpy(targets)

                gr_mask = beh_action_probs >= gr_safety_thresh

        if reward_column == 'reward':
            reward = np.asarray(data[reward_column])[-1]

            if normalize_reward and is_train:
                reward = (reward - train_reward_mu) / train_reward_std
            rewards[idx, T - 1] = reward
        else:

            raise Exception("We currrently do not offer training in this mode")

        eval_action_probs = eval_policy.get_action_probability(torch.from_numpy(features).float(), no_grad,
                                                               action_mask=gr_mask)

        pies[idx, :T] = torch.hstack([eval_action_probs[i, a] for i, a in enumerate(actions)])

    return pibs, pies, rewards, lengths.astype(int), MAX_TIME


def step(policy, batch, mse_loss=False, batch_mean=True):
    features = batch['features']
    targets = batch['targets']

    logp = policy(features)
    if batch_mean:
        loss = -torch.mean(torch.sum(targets * logp, dim=-1))
    else:
        loss = torch.sum(targets * logp, dim=-1)

    return loss


def evaluate(policy, val_dataloader, batch_mean=True):
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            val_loss = step(policy, batch, batch_mean=batch_mean)
            if batch_mean:
                losses.append(val_loss.detach().item())
            else:
                losses.append(val_loss.detach().numpy())

    if batch_mean:
        return np.mean(losses)
    else:
        loss_np = np.concatenate(losses)
        return - np.mean(loss_np)


def bc_train_policy(policy, train_df, valid_df, lr=1e-4, epochs=10, verbose=True, early_stop=False,
                    train_ess_early_stop=25, val_ess_early_stop=25, gr_safety_thresh=0.0,
                    return_weights=False, model_round_as_feature=False):
    if model_round_as_feature:
        import warnings
        warnings.warn('model_round_as_feature is on; not able to do PG training on this model')

    train_data = BCDataset(train_df, model_round_as_feature=model_round_as_feature)
    valid_data = BCDataset(valid_df, model_round_as_feature=model_round_as_feature)

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    optimizer = Adam(policy.parameters(), lr=lr)
    mean_train_losses, train_esss, mean_val_losses, val_esss = [], [], [], []
    train_weights, valid_weights = [], []
    train_opes, val_opes = [], []

    for e in range(epochs):
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = step(policy, batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())

        mean_train_loss = np.mean(train_losses)

        mean_val_loss = evaluate(policy, valid_loader, batch_mean=False)

        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(train_df, policy, no_grad=True,
                                                                                  is_train=True,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  model_round_as_feature=model_round_as_feature)
        train_wis, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time)

        train_opes.append(train_wis.item())
        train_weights.append(weights)

        train_ESS = 1 / (torch.sum(weights ** 2, axis=0))

        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(valid_df, policy, no_grad=True,
                                                                                  is_train=True,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  model_round_as_feature=model_round_as_feature)
        valid_wis, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time)

        valid_weights.append(weights)
        val_opes.append(valid_wis.item())

        val_ESS = 1 / (torch.sum(weights ** 2, axis=0))

        if verbose:
            print(
                "Epoch {} train loss: {:.3f}, train OPE: {:.3f}, train ESS: {:.3f}, val loss: {:.3f}, val OPE: {:.3f}, val ESS: {:.3f}".format(
                    e, mean_train_loss, train_wis, train_ESS, mean_val_loss, valid_wis, val_ESS
                ))

        mean_train_losses.append(mean_train_loss)
        mean_val_losses.append(mean_val_loss)
        train_esss.append(train_ESS.item())
        val_esss.append(val_ESS.item())

        if early_stop:
            if train_ESS >= train_ess_early_stop and val_ESS >= val_ess_early_stop:
                break

    if not return_weights:
        return mean_train_losses, train_esss, mean_val_losses, val_esss
    else:
        return mean_train_losses, train_esss, mean_val_losses, val_esss, train_weights, valid_weights, train_opes, val_opes


def ope_step(policy, df, ope_method, lambda_ess=4, gr_safety_thresh=0.0, is_train=True, return_weights=False,
             normalize_reward=False):
    pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                              no_grad=False,
                                                                              gr_safety_thresh=gr_safety_thresh,
                                                                              normalize_reward=normalize_reward)
    ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time)
    ess_theta = 1 / (torch.sum(weights ** 2, dim=0))

    loss = -(ope_score - (lambda_ess / torch.sqrt(ess_theta)))

    if not return_weights:
        return loss, ope_score.detach().item(), ess_theta.detach().item()
    else:
        return loss, ope_score.detach().item(), ess_theta.detach().item(), weights


def ope_evaluate(policy, df, ope_method, gr_safety_thresh, is_train=True, reward_column='reward', return_weights=False,
                 use_knn=False):
    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  reward_column=reward_column,
                                                                                  use_knn=use_knn)
        ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time)
        ess = 1 / (torch.sum(weights ** 2, dim=0))

    if return_weights:
        return ope_score, ess, weights
    else:
        return ope_score, ess


def th_to_np(np_array):
    return torch.from_numpy(np_array)


B = 100


def bca_bootstrap(pibs, pies, rewards, length, ope_method=wis_ope, alpha=0.05, max_time=MAX_TIME):
    pibs = pibs.numpy()
    pies = pies.numpy()
    rewards = rewards.numpy()

    n_users = pibs.shape[0]

    n_subsample = n_users

    wis_list = []
    for b in range(B):
        ids = np.random.choice(n_users, n_subsample)
        sam_rewards = rewards[ids, :]
        sam_pibs = pibs[ids, :]
        sam_pies = pies[ids, :]
        sam_length = length[ids]
        wis_pie, _ = ope_method(th_to_np(sam_pibs), th_to_np(sam_pies), th_to_np(sam_rewards), sam_length,
                                max_time=max_time)
        wis_list.append(wis_pie.numpy())
    y = []
    for i in range(n_users):
        sam_rewards = np.delete(rewards, i, axis=0)
        sam_pibs = np.delete(pibs, i, axis=0)
        sam_pies = np.delete(pies, i, axis=0)
        sam_length = np.delete(length, i, axis=0)
        wis_pie, _ = ope_method(th_to_np(sam_pibs), th_to_np(sam_pies), th_to_np(sam_rewards), sam_length,
                                max_time=max_time)
        y.append(wis_pie.numpy())

    wis_list = np.array(wis_list)
    wis_list = np.sort(wis_list)
    y = np.array(y)
    avg, _ = ope_method(th_to_np(pibs), th_to_np(pies), th_to_np(rewards), length, max_time=max_time)
    avg = avg.numpy()

    ql, qu = norm.ppf(alpha), norm.ppf(1 - alpha)

    num = np.sum((y.mean() - y) ** 3)
    den = 6 * np.sum((y.mean() - y) ** 2) ** 1.5
    ahat = num / den

    zhat = norm.ppf(np.mean(wis_list < avg))
    a1 = norm.cdf(zhat + (zhat + ql) / (1 - ahat * (zhat + ql)))
    a2 = norm.cdf(zhat + (zhat + qu) / (1 - ahat * (zhat + qu)))

    return np.quantile(wis_list, [a1, a2]), wis_list


def evaluate_policy_for_ci(policy, df, ope_method, gr_safety_thresh, alpha=0.05, is_train=True,
                           reward_column='reward', use_knn=False):
    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  reward_column=reward_column,
                                                                                  use_knn=use_knn)
        lb_ub, wis_list = bca_bootstrap(pibs, pies, rewards, lengths, alpha=alpha, ope_method=ope_method,
                                        max_time=max_time)

        avg, _ = ope_method(pibs, pies, rewards, lengths, max_time=max_time)

    return lb_ub, avg.item(), wis_list


def compute_ci_for_policies(bc_policy, df, ope_method, gr_safety_thresh, alpha=0.05, get_wis=False, is_train=True,
                            reward_column='reward', use_knn=False):
    lb_ub, wis, wis_list = evaluate_policy_for_ci(bc_policy, df, ope_method, gr_safety_thresh, alpha=alpha,
                                                  is_train=is_train,
                                                  reward_column=reward_column, use_knn=use_knn)
    score, ess = ope_evaluate(bc_policy, df, ope_method, gr_safety_thresh, is_train=is_train)

    if get_wis:
        return wis_list
    else:
        return wis, np.mean(wis_list), lb_ub, ess


class PGDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False):
        self.df = df
        self.model_round_as_feature = model_round_as_feature

        self.student_ids = self.df['user_id']

    def __len__(self):
        return self.student_ids.shape[0]

    def __getitem__(self, index):
        student_id = self.student_ids.iloc[index]

        return {'student_id': torch.from_numpy(np.array([student_id])).int()}


def offpolicy_pg_training(policy, train_df, valid_df, train_ope_method, folder_path, lr=1e-4, epochs=10, lambda_ess=4,
                          eval_ope_method=None, verbose=True, early_stop_ess=0., gr_safety_thresh=0.0,
                          return_weights=False,
                          use_knn=False, normalize_reward=False, device=None, is_full_mdp=False):
    eval_ope_method = train_ope_method if eval_ope_method is None else eval_ope_method

    optimizer = Adam(policy.parameters(), lr=lr)
    train_opes, train_esss, val_opes, val_esss = [], [], [], []
    train_losses = []

    train_weights, valid_weights = [], []

    ckpt_path = pjoin(folder_path, 'model.ckpt')

    best_epoch = 0
    for e in range(epochs):

        optimizer.zero_grad()

        loss, train_ope, train_ess = ope_step(policy, train_df, train_ope_method, lambda_ess,
                                              gr_safety_thresh, is_train=True, normalize_reward=normalize_reward)

        loss.backward()
        optimizer.step()

        val_ope, val_ess, weights = ope_evaluate(policy, valid_df, eval_ope_method, gr_safety_thresh, is_train=False,
                                                 return_weights=True, use_knn=use_knn)

        valid_weights.append(weights)

        train_opes.append(train_ope)
        train_esss.append(train_ess)

        val_opes.append(val_ope)
        val_esss.append(val_ess)

        train_losses.append(loss.cpu().detach().item())

        if verbose:
            print("Epoch {} train OPE Score: {:.2f}, ".format(e, train_ope) + \
                  "train loss: {:.2f}, train ESS: {:.2f}, val OPE Score: {:.2f}, val ESS: {:.2f}".format(
                      loss.detach().item(), train_ess, val_ope, val_ess))

        best_epoch = e

    best_val_ope = val_opes[best_epoch]

    best_valid_ess = val_esss[best_epoch]
    best_train_ope, best_train_ess = 0, 0
    train_losses = []

    if return_weights:
        return train_losses, train_opes, train_esss, val_opes, val_esss, valid_weights
    else:
        return train_losses, train_opes, train_esss, val_opes, val_esss, (
        best_train_ope, best_val_ope, best_train_ess, best_valid_ess)


def minibatch_offpolicy_pg_training(policy, train_df, valid_df, train_ope_method, folder_path, lr=1e-4, epochs=10,
                                    lambda_ess=4,
                                    eval_ope_method=None, verbose=True, early_stop_ess=0., gr_safety_thresh=0.0,
                                    return_weights=False,
                                    use_knn=False, normalize_reward=False, clip_lower=1e-16, clip_upper=1e2,
                                    batch_size=4):
    train_data = PGDataset(train_df)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,

        shuffle=True,
        num_workers=0
    )

    eval_ope_method = train_ope_method if eval_ope_method is None else eval_ope_method

    optimizer = Adam(policy.parameters(), lr=lr)
    train_opes, train_esss, val_opes, val_esss = [], [], [], []

    train_weights, valid_weights = [], []

    ckpt_path = pjoin(folder_path, 'model.ckpt')

    best_epoch = 0
    for e in range(epochs):

        train_losses = []
        train_opes = []
        train_ess = []

        for batch in train_loader:
            student_ids = batch['student_id'].squeeze(1).numpy().tolist()
            batch_df = train_df[train_df['user_id'].isin(student_ids)]

            optimizer.zero_grad()

            pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(batch_df, policy, is_train=True,
                                                                                      no_grad=False,
                                                                                      gr_safety_thresh=gr_safety_thresh,
                                                                                      normalize_reward=normalize_reward)

            ope_score, weights = clipped_is_ope(pibs, pies, rewards, lengths, max_time=max_time, clip_upper=clip_upper,
                                                clip_lower=clip_lower)
            normalized_weights = weights / weights.sum(dim=0)
            ess_theta = 1 / (torch.sum(normalized_weights ** 2, dim=0))

            loss = -(ope_score - (lambda_ess / torch.sqrt(ess_theta)))

            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().item())
            train_opes.append(ope_score.detach().item())
            train_ess.append(ess_theta.detach().item())

        val_ope, val_ess, weights = ope_evaluate(policy, valid_df, eval_ope_method, gr_safety_thresh, is_train=True,
                                                 return_weights=True, use_knn=use_knn)

        valid_weights.append(weights)

        val_opes.append(val_ope)
        val_esss.append(val_ess)

        if verbose:
            print("Epoch {} train OPE Score: {:.2f}, ".format(e, np.mean(train_opes)) + \
                  "train loss: {:.2f}, train ESS: {:.2f}, val OPE Score: {:.2f}, val ESS: {:.2f}".format(
                      np.mean(train_losses), np.mean(train_ess), val_ope, val_ess))

        torch.save(policy.state_dict(), ckpt_path)
        best_epoch = e

    if e > 0:
        policy.load_state_dict(torch.load(ckpt_path))

    best_val_ope = val_opes[best_epoch]

    best_valid_ess = val_esss[best_epoch]

    best_train_ope, best_train_ess = 0, 0
    train_losses = []

    if return_weights:
        return train_losses, train_opes, train_esss, val_opes, val_esss, valid_weights
    else:
        return train_losses, train_opes, train_esss, val_opes, val_esss, (
        best_train_ope, best_val_ope, best_train_ess, best_valid_ess)


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

from pois.offline_pg import set_random_seed, MLPPolicy, wis_ope, clipped_is_ope, cwpdis_ope, is_ope


class BCDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False, device=None, is_full_mdp=False):

        self.df = df
        self.model_round_as_feature = model_round_as_feature

        feature_names = ['hr_state', 'sysbp_state', 'oxygen_state',

                         'antibiotic_state', 'vaso_state', 'vent_state']

        if is_full_mdp:
            self.feature_names = full_feature_names
        else:
            self.feature_names = feature_names

        if model_round_as_feature:
            self.feature_names += ['model_round']

        self.target_names = ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]
        self.device = device

        self.features = torch.FloatTensor(df[self.feature_names].to_numpy()).to(device)
        self.targets = torch.FloatTensor(df[self.target_names].to_numpy()).to(device)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        return {'features': self.features[index],

                'targets': self.targets[index]}


feature_names = ['hr_state', 'sysbp_state', 'oxygen_state', 'antibiotic_state', 'vaso_state', 'vent_state']
target_names = ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]

primer_feature_names = ['hr_state', 'sysbp_state', 'oxygen_state', 'antibiotic_state', 'vaso_state', 'vent_state']

primer_target_names = ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]
full_feature_names = ["hr_state", "sysbp_state", "glucose_state", "oxygen_state",
                      "diabetes_idx", "antibiotic_state", "vaso_state", "vent_state"]

MAX_TIME = 20


def compute_is_weights_for_nn_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='Reward', no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                     model_round_as_feature=False,
                                     feat_names=None, act_name=None, sect_name=None, targ_names=None,
                                     device=None, is_deon_model=False, is_full_mdp=False):
    if is_full_mdp and feat_names is None:
        feature_names = full_feature_names
    else:
        feature_names = primer_feature_names if feat_names is None else feat_names

    section_name = 'Trajectory' if sect_name is None else sect_name
    target_names = primer_target_names if targ_names is None else targ_names

    action_name = 'Action_taken' if act_name is None else act_name

    df = behavior_df
    user_ids = df[section_name].unique()
    n = len(user_ids)

    MAX_TIME = max(behavior_df.groupby(section_name).size())

    pies = torch.zeros((n, MAX_TIME)).to(device)

    pibs = torch.zeros((n, MAX_TIME)).to(device)
    rewards = torch.zeros((n, MAX_TIME)).to(device)
    lengths = np.zeros((n))

    user_rewards = df.groupby(section_name)[reward_column].mean()
    train_reward_mu = user_rewards.mean()
    train_reward_std = user_rewards.std()

    for idx, user_id in enumerate(user_ids):
        data = df[df[section_name] == user_id]

        if not model_round_as_feature:
            features = np.asarray(data[feature_names]).astype(float)
        else:
            features = np.asarray(data[feature_names + ['model_round']]).astype(float)
        targets = np.asarray(data[target_names]).astype(float)
        actions = np.asarray(data[action_name]).astype(int)

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
                    knn_target_names = ""
                    knn_targets = np.asarray(data[knn_target_names]).astype(float)
                    assert knn_targets.shape[0] == targets.shape[0]
                    beh_action_probs = torch.from_numpy(knn_targets)

                    gr_mask = beh_action_probs >= gr_safety_thresh
            else:
                beh_action_probs = torch.from_numpy(targets)

                gr_mask = beh_action_probs >= gr_safety_thresh

        reward = np.asarray(data[reward_column])[-1]

        if normalize_reward and is_train:
            reward = (reward - train_reward_mu) / train_reward_std
        rewards[idx, T - 1] = reward

        if is_deon_model:
            eval_action_probs = eval_policy.softmax_prob(torch.from_numpy(features).float())
            pies[idx, :T] = torch.hstack([torch.FloatTensor([eval_action_probs[i, a]]) for i, a in enumerate(actions)])
        else:
            feats = torch.from_numpy(features).float().to(device)
            eval_action_probs = eval_policy.get_action_probability(feats, no_grad)
            pies[idx, :T] = torch.hstack([eval_action_probs[i, a] for i, a in enumerate(actions)])

    return pibs, pies, rewards, lengths.astype(int), MAX_TIME


def compute_is_weights_for_nn_policy_batched(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                             reward_column='Reward', no_grad=True,
                                             gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                             model_round_as_feature=False,
                                             feat_names=None, act_name=None, sect_name=None, targ_names=None,
                                             device=None):
    feature_names = primer_feature_names if feat_names is None else feat_names

    section_name = 'Trajectory' if sect_name is None else sect_name
    target_names = primer_target_names if targ_names is None else targ_names

    action_name = 'Action_taken' if act_name is None else act_name

    df = behavior_df
    user_ids = df[section_name].unique()
    n = len(user_ids)

    MAX_TIME = max(behavior_df.groupby(section_name).size())

    pies = torch.zeros((n, MAX_TIME)).to(device)

    pibs = torch.zeros((n, MAX_TIME)).to(device)
    rewards = torch.zeros((n, MAX_TIME)).to(device)
    lengths = np.zeros((n))

    user_rewards = df.groupby(section_name)[reward_column].mean()
    train_reward_mu = user_rewards.mean()
    train_reward_std = user_rewards.std()

    features = df[feature_names].to_numpy().astype(float)
    targets = df[target_names].to_numpy().astype(float)
    actions = df[action_name].to_numpy().astype(int)


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
                losses.append(val_loss.cpu().detach().item())
            else:
                losses.append(val_loss.cpu().detach().numpy())

    if batch_mean:
        return np.mean(losses)
    else:
        loss_np = np.concatenate(losses)
        return - np.mean(loss_np)


def bc_train_policy(policy, train_df, valid_df, lr=1e-4, epochs=10, verbose=True, early_stop=False,
                    train_ess_early_stop=25, val_ess_early_stop=25, gr_safety_thresh=0.0,
                    return_weights=False, model_round_as_feature=False, device=None, pretrain_mode=False,
                    is_full_mdp=False):
    if model_round_as_feature:
        import warnings
        warnings.warn('model_round_as_feature is on; not able to do PG training on this model')

    train_data = BCDataset(train_df, model_round_as_feature=model_round_as_feature, device=device,
                           is_full_mdp=is_full_mdp)
    valid_data = BCDataset(valid_df, model_round_as_feature=model_round_as_feature, device=device,
                           is_full_mdp=is_full_mdp)

    if device is None:
        train_loader = DataLoader(
            train_data,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=256,
            shuffle=True,
            num_workers=0,

        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=256,
            shuffle=False,
            num_workers=0,

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

        if not pretrain_mode:

            pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(train_df, policy, no_grad=True,
                                                                                      is_train=True,
                                                                                      gr_safety_thresh=gr_safety_thresh,
                                                                                      model_round_as_feature=model_round_as_feature,
                                                                                      device=device,
                                                                                      is_full_mdp=is_full_mdp)
            train_wis, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time, device=device)

            train_opes.append(train_wis.item())
            train_weights.append(weights)

            train_ESS = 1 / (torch.sum(weights ** 2, axis=0))

            pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(valid_df, policy, no_grad=True,
                                                                                      is_train=True,
                                                                                      gr_safety_thresh=gr_safety_thresh,
                                                                                      model_round_as_feature=model_round_as_feature,
                                                                                      device=device,
                                                                                      is_full_mdp=is_full_mdp)
            valid_wis, weights = wis_ope(pibs, pies, rewards, lengths, max_time=max_time, device=device)

            valid_weights.append(weights.cpu().detach())
            val_opes.append(valid_wis.item())

            val_ESS = 1 / (torch.sum(weights ** 2, axis=0))
        else:
            train_wis, train_ESS, valid_wis, val_ESS = 0, torch.zeros(1), 0, torch.zeros(1)

        if verbose:
            print(
                "Epoch {} train loss: {:.3f}, train OPE: {:.3f}, train ESS: {:.3f}, val loss: {:.3f}, val OPE: {:.3f}, val ESS: {:.3f}".format(
                    e, mean_train_loss, train_wis, train_ESS, mean_val_loss, valid_wis, val_ESS))

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
             normalize_reward=False, device=None, is_full_mdp=False):
    pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                              no_grad=False,
                                                                              gr_safety_thresh=gr_safety_thresh,
                                                                              normalize_reward=normalize_reward,
                                                                              device=device, is_full_mdp=is_full_mdp)
    ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time, device=device)
    ess_theta = 1 / (torch.sum(weights ** 2, dim=0))

    loss = -(ope_score - (lambda_ess / torch.sqrt(ess_theta)))

    if not return_weights:
        return loss, ope_score.cpu().detach().item(), ess_theta.cpu().detach().item()
    else:
        return loss, ope_score.cpu().detach().item(), ess_theta.cpu().detach().item(), weights


def ope_evaluate(policy, df, ope_method, gr_safety_thresh, is_train=True, reward_column='Reward', feat_names=None,
                 return_weights=False,
                 use_knn=False, device=None, is_full_mdp=False):
    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  reward_column=reward_column,
                                                                                  use_knn=use_knn, device=device,
                                                                                  is_full_mdp=is_full_mdp,
                                                                                  feat_names=feat_names)
        ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time, device=device)
        ess = 1 / (torch.sum(weights ** 2, dim=0))

    if return_weights:
        return ope_score, ess, weights
    else:
        return ope_score, ess


def save_policy_to_csv(bc_pg_policy, dataset_size, model_size, split, alg='bc', is_full_data=False):
    df = pd.read_csv("./sontag_sepsis_folder/all_states.csv")
    features = df[feature_names]
    features = features.to_numpy().astype(float)
    features = torch.from_numpy(features).float()

    logp = bc_pg_policy(features)
    action_probs = torch.exp(logp)
    action_taken = torch.max(logp, dim=1)[1]
    df['Action_taken'] = action_taken
    for i in range(8):
        df[f'beh_p_{i}'] = action_probs[:, i].detach().numpy()

    if not is_full_data:
        df.to_csv("./sontag_sepsis_xtreme_difficulty/all_states_with_size_{}_{}_model_{}_split_{}.csv".format(
            dataset_size, alg, "_".join([str(c) for c in model_size]), split
        ), index=False)
    else:
        df.to_csv("./sontag_sepsis_xtreme_difficulty/all_states_with_size_{}_{}_model_{}_full_data.csv".format(
            dataset_size, alg, "_".join([str(c) for c in model_size])
        ), index=False)


import os


def save_policy_to_csv_sweep(bc_pg_policy, dataset_size, alg_name, split, feature_names, exp_name, is_full_data=False,
                             cv=False, num_fold=10, device=None, save_folder="sontag_sepsis_ijcai_sweep_results"):
    fold_or_split = 'split' if not cv else f"{num_fold}fold"

    df = pd.read_csv("./sontag_sepsis_folder/all_states.csv")
    features = df[feature_names]
    features = features.to_numpy().astype(float)
    features = torch.from_numpy(features).float().to(device)

    logp = bc_pg_policy(features).to('cpu')
    action_probs = torch.exp(logp)
    action_probs = action_probs.detach().numpy()
    action_taken = torch.max(logp, dim=1)[1]
    df['Action_taken'] = action_taken
    for i in range(8):
        df[f'beh_p_{i}'] = action_probs[:, i]

    os.makedirs("./{}/{}".format(save_folder, alg_name), exist_ok=True)

    if not is_full_data:
        file_name = "./{}/{}/all_states_with_size_{}_exp_{}_{}_{}.csv".format(
            save_folder, alg_name, dataset_size, exp_name, fold_or_split, split
        )
        df.to_csv(file_name, index=False)
    else:
        file_name = "./{}/{}/all_states_with_size_{}_exp_{}_{}_{}_full_data.csv".format(
            save_folder, alg_name, dataset_size, exp_name, fold_or_split, split
        )
        df.to_csv(file_name, index=False)

    return file_name


def th_to_np(np_array):
    return torch.from_numpy(np_array)


B = 100


def bca_bootstrap(pibs, pies, rewards, length, ope_method=wis_ope, alpha=0.05, max_time=MAX_TIME, device=None):
    pibs = pibs.cpu().numpy()
    pies = pies.cpu().numpy()
    rewards = rewards.cpu().numpy()

    n_users = pibs.shape[0]

    n_subsample = n_users

    wis_list = []
    for b in range(B):
        ids = np.random.choice(n_users, n_subsample)
        sam_rewards = rewards[ids, :]
        sam_pibs = pibs[ids, :]
        sam_pies = pies[ids, :]
        sam_length = length[ids]
        wis_pie, _ = ope_method(th_to_np(sam_pibs).to(device), th_to_np(sam_pies).to(device),
                                th_to_np(sam_rewards).to(device), sam_length,
                                max_time=max_time, device=device)
        wis_list.append(wis_pie.cpu().numpy())
    y = []
    for i in range(n_users):
        sam_rewards = np.delete(rewards, i, axis=0)
        sam_pibs = np.delete(pibs, i, axis=0)
        sam_pies = np.delete(pies, i, axis=0)
        sam_length = np.delete(length, i, axis=0)
        wis_pie, _ = ope_method(th_to_np(sam_pibs).to(device), th_to_np(sam_pies).to(device),
                                th_to_np(sam_rewards).to(device), sam_length,
                                max_time=max_time, device=device)
        y.append(wis_pie.cpu().numpy())

    wis_list = np.array(wis_list)
    wis_list = np.sort(wis_list)
    y = np.array(y)
    avg, _ = ope_method(th_to_np(pibs).to(device), th_to_np(pies).to(device), th_to_np(rewards).to(device), length,
                        max_time=max_time, device=device)
    avg = avg.cpu().numpy()

    ql, qu = norm.ppf(alpha), norm.ppf(1 - alpha)

    num = np.sum((y.mean() - y) ** 3)
    den = 6 * np.sum((y.mean() - y) ** 2) ** 1.5
    ahat = num / den

    zhat = norm.ppf(np.mean(wis_list < avg))
    a1 = norm.cdf(zhat + (zhat + ql) / (1 - ahat * (zhat + ql)))
    a2 = norm.cdf(zhat + (zhat + qu) / (1 - ahat * (zhat + qu)))

    return np.quantile(wis_list, [a1, a2]), wis_list


def evaluate_policy_for_ci(policy, df, ope_method, gr_safety_thresh, alpha=0.05, is_train=True,
                           reward_column='Reward', use_knn=False):
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
                            reward_column='Reward', use_knn=False):
    lb_ub, wis, wis_list = evaluate_policy_for_ci(bc_policy, df, ope_method, gr_safety_thresh, alpha=alpha,
                                                  is_train=is_train,
                                                  reward_column=reward_column, use_knn=use_knn)
    score, ess = ope_evaluate(bc_policy, df, ope_method, gr_safety_thresh, is_train=is_train)

    if get_wis:
        return wis_list
    else:
        return wis, np.mean(wis_list), lb_ub, ess


def evaluate_policy_for_ci_sweep(policy, df, ope_method, gr_safety_thresh, alpha=0.05, is_train=True,
                                 reward_column='Reward', use_knn=False, feat_names=None, act_name=None, sect_name=None,
                                 targ_names=None, is_deon_model=False, device=None):
    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  reward_column=reward_column,
                                                                                  use_knn=use_knn,
                                                                                  feat_names=feat_names,
                                                                                  sect_name=sect_name,
                                                                                  act_name=act_name,
                                                                                  targ_names=targ_names,
                                                                                  is_deon_model=is_deon_model,
                                                                                  device=device)

        lb_ub, wis_list = bca_bootstrap(pibs, pies, rewards, lengths, alpha=alpha, ope_method=ope_method,
                                        max_time=max_time, device=device)

        avg, _ = ope_method(pibs, pies, rewards, lengths, max_time=max_time, device=device)

    return lb_ub, avg.item(), wis_list


class PGDataset(Dataset):
    def __init__(self, df, model_round_as_feature=False):
        self.df = df
        self.model_round_as_feature = model_round_as_feature

        self.student_ids = self.df['Trajectory']

    def __len__(self):
        return self.student_ids.shape[0]

    def __getitem__(self, index):
        student_id = self.student_ids.iloc[index]

        return {'student_id': torch.from_numpy(np.array([student_id])).int()}


def minibatch_offpolicy_pg_training(policy, train_df, valid_df, train_ope_method, folder_path, lr=1e-4, epochs=10,
                                    lambda_ess=4,
                                    eval_ope_method=None, verbose=True, early_stop_ess=0., gr_safety_thresh=0.0,
                                    return_weights=False,
                                    use_knn=False, normalize_reward=False, clip_lower=1e-16, clip_upper=1e2,
                                    batch_size=4, device=None, sweep_mode=False, is_full_mdp=False):
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
            batch_df = train_df[train_df['Trajectory'].isin(student_ids)]

            optimizer.zero_grad()

            pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(batch_df, policy, is_train=True,
                                                                                      no_grad=False,
                                                                                      gr_safety_thresh=gr_safety_thresh,
                                                                                      normalize_reward=normalize_reward,
                                                                                      device=device,
                                                                                      is_full_mdp=is_full_mdp)

            ope_score, weights = clipped_is_ope(pibs, pies, rewards, lengths, max_time=max_time, clip_upper=clip_upper,
                                                clip_lower=clip_lower, device=device)
            normalized_weights = weights / weights.sum(dim=0)
            ess_theta = 1 / (torch.sum(normalized_weights ** 2, dim=0))

            loss = -(ope_score - (lambda_ess / torch.sqrt(ess_theta)))

            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().detach().item())
            train_opes.append(ope_score.cpu().detach().item())
            train_ess.append(ess_theta.cpu().detach().item())

        if not sweep_mode:
            val_ope, val_ess, weights = ope_evaluate(policy, valid_df, eval_ope_method, gr_safety_thresh, is_train=True,
                                                     return_weights=True, use_knn=use_knn, device=device,
                                                     is_full_mdp=is_full_mdp)

            valid_weights.append(weights)

            val_opes.append(val_ope)
            val_esss.append(val_ess)
        else:
            val_ope, val_ess = 0, 0
            val_opes.append(val_ope)
            val_esss.append(val_ess)

        if verbose:
            print("Epoch {} train OPE Score: {:.2f}, ".format(e, np.mean(train_opes)) + \
                  "train loss: {:.2f}, train ESS: {:.2f}, val OPE Score: {:.2f}, val ESS: {:.2f}".format(
                      np.mean(train_losses), np.mean(train_ess), val_ope, val_ess))

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
                                              gr_safety_thresh, is_train=True, normalize_reward=normalize_reward,
                                              device=device, is_full_mdp=is_full_mdp)

        loss.backward()
        optimizer.step()

        val_ope, val_ess, weights = ope_evaluate(policy, valid_df, eval_ope_method, gr_safety_thresh, is_train=False,
                                                 return_weights=True, use_knn=use_knn, device=device,
                                                 is_full_mdp=is_full_mdp)

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


if __name__ == '__main__':

    for dataset_size in [200, 1000, 5000]:

        for model_size in [[64, 64]]:

            set_random_seed(72)

            print("config: dataset {}, method {}, model {}".format(dataset_size, 'bc', model_size))
            train_df = pd.read_csv(f"./sontag_sepsis_folder/sepsis_{dataset_size}.csv")
            valid_df = pd.read_csv(f"./sontag_sepsis_folder/sepsis_{dataset_size}.csv")

            train_df = train_df[train_df['Action_taken'] != -1]
            valid_df = valid_df[valid_df['Action_taken'] != -1]

            bc_pg_policy = MLPPolicy([6] + model_size + [8])
            bc_train_losses, bc_train_esss, bc_val_losses, bc_val_esss, train_weights, valid_weights, train_opes, val_opes = bc_train_policy(
                bc_pg_policy, train_df,
                valid_df,
                epochs=5,

                lr=1e-3,
                verbose=True, early_stop=True,
                train_ess_early_stop=25, val_ess_early_stop=15,
                return_weights=True,
                model_round_as_feature=False)

            df = pd.read_csv("./sontag_sepsis_folder/all_states.csv")
            features = df[feature_names]
            features = features.to_numpy().astype(float)
            features = torch.from_numpy(features).float()

            logp = bc_pg_policy(features)
            action_probs = torch.exp(logp)
            action_taken = torch.max(logp, dim=1)[1]
            df['Action_taken'] = action_taken
            for i in range(8):
                df[f'beh_p_{i}'] = action_probs[:, i].detach().numpy()

            df.to_csv("./sontag_sepsis_folder/all_states_with_size_{}_{}_model_{}.csv".format(
                dataset_size, 'bc', "_".join([str(c) for c in model_size])
            ), index=False)

            print("config: dataset {}, method {}, model {}".format(dataset_size, 'bcpg', model_size))

            train_losses, train_opes, train_esss, val_opes, val_esss, pg_valid_weights = minibatch_offpolicy_pg_training(
                bc_pg_policy, train_df, valid_df,
                wis_ope, "./tests/", lr=3e-4,
                epochs=5,
                lambda_ess=0.01,
                gr_safety_thresh=0.01,
                verbose=True,
                early_stop_ess=0,
                return_weights=True, clip_upper=1e2)

            logp = bc_pg_policy(features)
            action_probs = torch.exp(logp)
            action_taken = torch.max(logp, dim=1)[1]
            df['Action_taken'] = action_taken
            for i in range(8):
                df[f'beh_p_{i}'] = action_probs[:, i].detach().numpy()

            df.to_csv("./sontag_sepsis_folder/all_states_with_size_{}_{}_model_{}.csv".format(
                dataset_size, 'bcpg', "_".join([str(c) for c in model_size])
            ), index=False)

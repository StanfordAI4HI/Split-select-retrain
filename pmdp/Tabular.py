import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from OPE import ope_evaluate, wis_ope
import os


class config:
    feature_names = ["hr_state", "sysbp_state", "oxygen_state", "antibiotic_state", "vaso_state", "vent_state"]
    feature_range = [3, 3, 2, 2, 2, 2]

    target_names = [f"beh_p_{i}" for i in range(8)]
    action_name = "Action_taken"
    reward_name = "Reward"
    section_name = "Trajectory"


def getEndState():
    end_state = [0 for _ in range(len(config.feature_range))]
    end_state[-1] = config.feature_range[-1]
    return end_state


def get_IndexArray(index_a, array):
    """
    Arguments:
    - index_a: (N, D) numpy array of all N possibilities for D-dimensional vector
    - array:   (B, D) numpy array

    Returns:
    - indices B-dimensional array of the index each row of (param) array appears in (param) index_a
    """
    return np.where(np.prod(index_a[None] == array[:, None], -1), np.arange(len(index_a)), 0).sum(1)


def getAllFeatures(remaining_features):
    if len(remaining_features) == 0:
        return [[]]
    rest_features = getAllFeatures(remaining_features[1:])
    all_features = []
    for i in range(remaining_features[0]):
        for each in rest_features:
            all_features.append([i] + each)
    return all_features


def getDfInfo(df):
    """
    Returns:
    - all possibile states, state-action pairs, rewards and the dataframe of a dataset
    """
    feature_name = config.feature_names
    allstates = getAllFeatures(config.feature_range)
    allstates += [getEndState()]
    allstates = np.array(allstates).astype(int)
    all_rew = np.array([[-1, 0, 1]]).T.astype(int)
    total_states = len(allstates)
    rep_allstates = np.repeat(allstates, 8, axis=0)
    rep_allactions = np.repeat(np.arange(8)[None], total_states, 0).reshape(-1, 1)
    all_sa = np.concatenate([rep_allstates, rep_allactions], axis=-1)
    return allstates, all_sa, all_rew, df


def get_MLE_model(allstates, all_sa, all_rew, df):
    """
    counts the all (s, a, ns) pairs in the dataset, and creates the tabular MLE dynamics model
    """
    feature_name = config.feature_names
    traj = np.asarray(df["Trajectory"]).astype(int)
    r = np.asarray(df["Reward"]).astype(int)[:, None]
    states = np.asarray(df[feature_name]).astype(float)
    state_action = np.asarray(df[feature_name + ["Action_taken"]]).astype(int)
    sa = state_action[:-1][traj[:-1] == traj[1:]]
    ns = states[1:][traj[:-1] == traj[1:]]
    end_sa = np.concatenate([state_action[:-1][traj[:-1] != traj[1:]], state_action[-1:]])
    end_r = np.concatenate([r[:-1][traj[:-1] != traj[1:]], r[-1:]])
    end_state = getEndState()

    all_sa_index = get_IndexArray(all_sa, state_action)
    sa_index = get_IndexArray(all_sa, sa)
    ns_index = get_IndexArray(allstates, ns)
    r_index = get_IndexArray(all_rew, r)
    end_sa_index = get_IndexArray(all_sa, end_sa)
    end_s_index = get_IndexArray(allstates, np.array([end_state for _ in range(len(end_sa_index))]))
    pairs, counts = np.unique(np.array([sa_index, ns_index]), axis=1, return_counts=True)
    e_pairs, e_counts = np.unique(np.array([end_sa_index, end_s_index]), axis=1, return_counts=True)
    r_pairs, r_counts = np.unique(np.array([all_sa_index, r_index]), axis=1, return_counts=True)
    d_pairs, d_counts = np.unique(np.array([end_sa_index, [1] * len(end_sa_index)]), axis=1, return_counts=True)
    nd_pairs, nd_counts = np.unique(np.array([sa_index, [0] * len(sa_index)]), axis=1, return_counts=True)
    unique_sa, sa_counts = np.unique(state_action, axis=0, return_counts=True)
    u_sa_index = get_IndexArray(all_sa, unique_sa)

    T = np.zeros((len(all_sa), len(allstates)))
    rew = np.zeros((len(all_sa), 3))
    done = np.zeros((len(all_sa), 2))

    T[pairs[0], pairs[1]] = counts
    T[e_pairs[0], e_pairs[1]] = e_counts
    T[-8:, -1] = 1
    rew[r_pairs[0], r_pairs[1]] = r_counts
    rew[-8:, 1] = 1
    done[d_pairs[0], d_pairs[1]] = d_counts
    done[nd_pairs[0], nd_pairs[1]] = nd_counts
    done[-8:, 1] = 1

    T_denom, R_denom, D_denom = T.sum(-1, keepdims=True), rew.sum(-1, keepdims=True), done.sum(-1, keepdims=True)
    T = T / np.where(T_denom == 0, 1, T_denom)
    rew = rew / np.where(R_denom == 0, 1, R_denom)
    done = done / np.where(D_denom == 0, 1, D_denom)
    n_sa = np.zeros(len(all_sa))
    n_sa[u_sa_index] = sa_counts
    n_sa[-8:] = 1e8
    return T, rew, done, n_sa


def trainDynamics(df, allstates, all_sa, all_rew, iterations, mdp_numbers):
    """
    samples (s, a, r, ns) tuples from the dataset and creates a bootstrapped MLE estimate of the dynamics
    Returns:
    - T, R : the dynamics and rewards model respectively
    """
    sarns = []
    for traj_i in df["Trajectory"].unique():
        data = df[df["Trajectory"] == traj_i]
        sa = np.asarray(data[config.feature_names + ["Action_taken"]]).astype(int)
        ns = np.concatenate([np.asarray(data[config.feature_names])[1:], np.array([getEndState()])]).astype(int)
        r = np.asarray(data[["Reward"]]).astype(int)
        sa_index = get_IndexArray(all_sa, sa)[:, None]
        ns_index = get_IndexArray(allstates, ns)[:, None]
        r_index = get_IndexArray(all_rew, r)[:, None]
        sarns.append(np.concatenate([sa_index, r_index, ns_index], -1))
    sarns.append(np.array([[len(all_sa) - i - 1, 1, len(allstates) - 1] for i in range(8)]))
    sarns = np.concatenate(sarns)
    T = np.zeros((mdp_numbers, len(all_sa), len(allstates)))

    R = np.zeros((mdp_numbers, len(all_sa), 3))

    ensemble = np.repeat(np.arange(mdp_numbers), 64)

    for i in range(iterations):
        indices = np.random.choice(len(sarns), mdp_numbers * 64)

        batch = sarns[indices]
        T_pairs, T_counts = np.unique(np.array([ensemble, batch[:, 0], batch[:, 2]]), axis=1, return_counts=True)
        R_pairs, R_counts = np.unique(np.array([ensemble, batch[:, 0], batch[:, 1]]), axis=1, return_counts=True)
        T[T_pairs[0], T_pairs[1], T_pairs[2]] += T_counts
        R[R_pairs[0], R_pairs[1], R_pairs[2]] += R_counts
    T_denom = T.sum(-1, keepdims=True)
    R_denom = R.sum(-1, keepdims=True)
    T = T / np.where(T_denom == 0, 1, T_denom)
    R = R / np.where(R_denom == 0, 1, R_denom)
    return T, R


def trainEnsemble(df, allstates, all_sa, all_rew, iterations, mdp_numbers):
    return trainDynamics(df, allstates, all_sa, all_rew, iterations, mdp_numbers)


def get_policy(t, r, allstates, all_sa, all_rew, n_sa, useEnsemble, penalty_coef=.1, temperature=1, mdp_numbers=7):
    V = np.zeros(len(allstates))
    Q = np.zeros((len(allstates), 8))
    total_iter = 0
    dist = 1
    gamma = .99
    pi = None
    delta = .01
    while dist > 1e-6 and total_iter < 600:
        total_iter += 1
        if useEnsemble:
            choice = np.random.choice(mdp_numbers)

            indices = [choice for _ in range(len(all_sa))]
            T = t[indices, np.arange(len(all_sa))]
            R = r[indices, np.arange(len(all_sa))]
            error = penalty_coef * np.sqrt(2 * np.log(1 / delta) / n_sa).reshape((len(allstates), 8))
        else:
            T = t
            R = r
            error = penalty_coef * np.sqrt(2 * np.log(1 / delta) / n_sa).reshape((len(allstates), 8))
        probs = T.reshape((len(allstates), 8, -1))
        exp_r = (R * all_rew.T).sum(1).reshape((len(allstates), 8))
        pessimistic_r = np.clip(exp_r - error, -1, 1)
        Q = pessimistic_r + gamma * (probs * V).sum(-1)
        V_new = np.max(Q, 1)
        pi = np.argmax(Q, 1)
        dist = abs(V - V_new).max()
        V = V_new
    soft_pi = F.softmax(torch.tensor(Q).float() / temperature, dim=-1)

    return pi, soft_pi


class dummyPolicy(object):
    def __init__(self, s_to_a, s_to_ind):
        self.setParams(s_to_a, s_to_ind, None)

    def setParams(self, s_to_a, s_to_ind, soft_pi):
        self.s_to_ind = s_to_ind
        self.s_to_a = s_to_a
        self.soft_pi = soft_pi

    def softmax_prob(self, features):
        features = features.cpu().numpy().astype(int)
        s = get_IndexArray(self.s_to_ind, features)
        return self.soft_pi[s]


def train_MLE_tabular(policy, train_df, valid_df, parameters):
    allstates, all_sa, all_rew, _ = getDfInfo(train_df)
    T, rew, done, n_sa = get_MLE_model(allstates, all_sa, all_rew, train_df)
    All_T, All_R = trainEnsemble(train_df, allstates, all_sa, all_rew, parameters.iterations)
    pi, soft_pi = get_policy(T, rew, allstates, all_sa, all_rew, n_sa + 1e-8, False, parameters.penalty_coef,
                             parameters.temperature, )
    policy.setParams(pi, allstates, soft_pi)
    ope_score, ess = ope_evaluate(policy, valid_df, wis_ope, 0, reward_column=parameters.reward_name,
                                  feat_names=parameters.feature_names, act_name=parameters.action_name,
                                  sect_name=parameters.trajectory_name, targ_names=parameters.target_names, )
    return ope_score, ess


def train_ensemble_tabular(policy, train_df, valid_df, parameters):
    allstates, all_sa, all_rew, _ = getDfInfo(train_df)
    T, rew, done, n_sa = get_MLE_model(allstates, all_sa, all_rew, train_df)
    All_T, All_R = trainEnsemble(train_df, allstates, all_sa, all_rew, parameters.iterations, parameters.mdp_numbers)
    pi, soft_pi = get_policy(All_T, All_R, allstates, all_sa, all_rew, n_sa + 1e-8, True, parameters.penalty_coef,
                             parameters.temperature, parameters.mdp_numbers)
    policy.setParams(pi, allstates, soft_pi)
    ope_score, ess = ope_evaluate(policy, valid_df, wis_ope, 0, reward_column=parameters.reward_name,
                                  feat_names=parameters.feature_names, act_name=parameters.action_name,
                                  sect_name=parameters.trajectory_name, targ_names=parameters.target_names, )
    return ope_score, ess


target_names = ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]


def get_mdp_policy(filename):
    bcpg_policy_pd = pd.read_csv(filename)
    bcpg_policy = bcpg_policy_pd[target_names].to_numpy()

    return bcpg_policy


def save_policy_to_csv(policy, dataset_size, model_size, split, feature_names, alg='alg', is_full_data=False):
    df = pd.read_csv("./sontag_sepsis_folder/all_states.csv")
    features = df[feature_names]
    features = features.to_numpy().astype(float)
    features = torch.from_numpy(features).float()

    action_probs = policy.softmax_prob(features)
    action_taken = np.argmax(action_probs, axis=1)
    df['Action_taken'] = action_taken
    for i in range(8):
        df[f'beh_p_{i}'] = action_probs[:, i]

    if not is_full_data:
        df.to_csv("./sontag_sepsis_ijcai_results/all_states_with_size_{}_{}_model_{}_split_{}.csv".format(
            dataset_size, alg, "_".join([str(c) for c in model_size]), split
        ), index=False)
    else:
        df.to_csv("./sontag_sepsis_ijcai_results/all_states_with_size_{}_{}_model_{}_full_data.csv".format(
            dataset_size, alg, "_".join([str(c) for c in model_size])
        ), index=False)


def save_policy_to_csv_sweep(policy, dataset_size, alg_name, split, feature_names, exp_name, is_full_data=False,
                             cv=False, num_fold=10, save_folder="sontag_sepsis_ijcai_sweep_results"):
    fold_or_split = 'split' if not cv else f"{num_fold}fold"

    df = pd.read_csv("./sontag_sepsis_folder/all_states.csv")
    features = df[feature_names]
    features = features.to_numpy().astype(float)
    features = torch.from_numpy(features).float()

    action_probs = policy.softmax_prob(features)
    action_taken = np.argmax(action_probs, axis=1)
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

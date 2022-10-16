import numpy as np
import torch
from sepsisSimDiabetes.State import State

feature_names = ["hr_state", "sysbp_state", "glucose_state",
                 "antibiotic_state", "vaso_state", "vent_state"]

target_names = ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]

MAX_TIME = 28


def compute_is_weights_for_mdp_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                      reward_column='adjusted_score', no_grad=True,
                                      gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                      model_round_as_feature=False):
    df = behavior_df
    user_ids = df['Trajectory'].unique()
    n = len(user_ids)

    MAX_TIME = max(behavior_df.groupby('Trajectory').size())

    pies = torch.zeros((n, MAX_TIME))

    pibs = torch.zeros((n, MAX_TIME))
    rewards = torch.zeros((n, MAX_TIME))
    lengths = np.zeros((n))

    user_rewards = df.groupby("Trajectory")[reward_column].mean()
    train_reward_mu = user_rewards.mean()
    train_reward_std = user_rewards.std()

    for idx, user_id in enumerate(user_ids):
        data = df[df['Trajectory'] == user_id]

        if not model_round_as_feature:
            features = np.asarray(data[feature_names]).astype(float)

            features_idx_list = []
            for feature_idx in data['State_id']:
                this_state = State(state_idx=feature_idx, idx_type='full',
                                   diabetic_idx=1)

                features_idx_list.append(this_state.get_state_idx('proj_obs'))

            features_idxs = np.array(features_idx_list).astype(int)
        else:
            features = np.asarray(data[feature_names + ['model_round']]).astype(float)
        targets = np.asarray(data[target_names]).astype(float)
        actions = np.asarray(data['Action_taken']).astype(int)

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
                beh_action_probs = torch.from_numpy(targets)

                gr_mask = beh_action_probs >= gr_safety_thresh

        if reward_column == 'Reward':
            reward = max(np.asarray(data[reward_column]))
            if reward == 0:
                reward = min(np.asarray(data[reward_column]))

            if normalize_reward and is_train:
                reward = (reward - train_reward_mu) / train_reward_std

            rewards[idx, T - 1] = reward
        else:

            raise Exception("We currrently do not offer training in this mode")

        eval_action_probs = torch.from_numpy(eval_policy[features_idxs, :])
        pies[idx, :T] = torch.hstack([eval_action_probs[i, a] for i, a in enumerate(actions)])

    return pibs, pies, rewards, lengths.astype(int), MAX_TIME


def wis_ope(pibs, pies, rewards, length, no_weight_norm=False, max_time=MAX_TIME, per_sample=False, clip_lower=1e-16,
            clip_upper=1e3):
    n = pibs.shape[0]
    weights = torch.ones((n, MAX_TIME))

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0
            last = last * (pies[i, t] / pibs[i, t])
            weights[i, t] = last
        weights[i, length[i]:] = weights[i, length[i] - 1]

    weights = torch.clip(weights, clip_lower, clip_upper)
    if not no_weight_norm:
        weights_norm = weights.sum(dim=0)
        weights /= weights_norm

    else:
        weights /= n

    if not per_sample:
        return (weights[:, -1] * rewards.sum(dim=-1)).sum(dim=0), weights[:, -1]
    else:

        return weights[:, -1] * rewards.sum(dim=-1), weights[:, -1]

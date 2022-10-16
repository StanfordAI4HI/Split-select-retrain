import numpy as np
import torch

primer_feature_names = ['stage_norm', 'failed_attempts_norm', 'pos_norm', 'neg_norm',
                        'hel_norm', 'anxiety_norm', 'grade_norm', 'pre', 'anxiety']
primer_categorical_features = ['stage']

primer_target_names = ["p_hint", "p_nothing", "p_encourage", "p_question"]

primer_feature_names = primer_feature_names + primer_categorical_features

MAX_TIME = 28


def convertState(states):
    states = states.astype(int)
    b = states.shape[0]
    one = np.zeros((b, 3))
    two = np.zeros((b, 3))
    three = np.zeros((b, 5))
    four = np.zeros((b, 2))
    five = np.zeros((b, 2))
    six = np.zeros((b, 3))
    one[np.arange(b), states[:, 0]] = 1
    two[np.arange(b), states[:, 1]] = 1
    three[np.arange(b), states[:, 2]] = 1
    four[np.arange(b), states[:, 3]] = 1
    five[np.arange(b), states[:, 4]] = 1
    six[np.arange(b), states[:, 5]] = 1
    end_state = np.zeros(18)
    end_state[-1] = 1
    new_states = np.where(states[:, 5:] == 2, end_state, np.concatenate([one, two, three, four, five, six], axis=1))
    return new_states


def compute_is_weights_for_nn_policy(behavior_df, eval_policy, eps=0.05, temp=0.1,
                                     reward_column='adjusted_score', feat_names=None, act_name=None, sect_name=None,
                                     targ_names=None, no_grad=True,
                                     gr_safety_thresh=0.0, is_train=True, normalize_reward=False, use_knn=False,
                                     model_round_as_feature=False):
    feature_names = primer_feature_names if feat_names is None else feat_names
    section_name = 'user_id' if sect_name is None else sect_name
    target_names = primer_target_names if targ_names is None else targ_names
    action_name = 'action' if act_name is None else act_name
    df = behavior_df
    user_ids = df[section_name].unique()
    n = len(user_ids)

    MAX_TIME = max(behavior_df.groupby(section_name).size())

    pies = torch.zeros((n, MAX_TIME))

    pibs = torch.zeros((n, MAX_TIME))
    rewards = torch.zeros((n, MAX_TIME))
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

        eval_action_probs = eval_policy.softmax_prob(torch.from_numpy(features).float())
        pies[idx, :T] = torch.hstack([torch.FloatTensor([eval_action_probs[i, a]]) for i, a in enumerate(actions)])

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


def ope_evaluate(policy, df, ope_method, gr_safety_thresh, is_train=True, reward_column='adjusted_score',
                 feat_names=None, act_name=None, sect_name=None, targ_names=None, return_weights=False, use_knn=False):
    with torch.no_grad():
        pibs, pies, rewards, lengths, max_time = compute_is_weights_for_nn_policy(df, policy, is_train=is_train,
                                                                                  gr_safety_thresh=gr_safety_thresh,
                                                                                  reward_column=reward_column,
                                                                                  feat_names=feat_names,
                                                                                  sect_name=sect_name,
                                                                                  act_name=act_name,
                                                                                  targ_names=targ_names,
                                                                                  use_knn=use_knn)
        ope_score, weights = ope_method(pibs, pies, rewards, lengths, max_time=max_time)
        ess = 1 / (torch.sum(weights ** 2, dim=0))

    if return_weights:
        return ope_score, ess, weights
    else:
        return ope_score, ess

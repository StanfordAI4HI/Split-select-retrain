import numpy as np
from BvftUtil import BvftRecord


class BVFT(object):
    def __init__(self, q_functions, data, gamma, rmax, rmin, record: BvftRecord = BvftRecord(), q_type='tabular',
                 verbose=False, bins=None, data_size=500):
        self.data = data
        self.gamma = gamma
        self.res = 0
        self.q_sa_discrete = []
        self.q_to_data_map = []
        self.q_size = len(q_functions)
        self.verbose = verbose
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100]

        self.bins = bins
        self.q_sa = []
        self.r_plus_vfsp = []
        self.q_functions = q_functions
        self.record = record

        if q_type == 'tabular':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            states = np.array([t[0] for t in self.data])
            for Q in q_functions:
                self.q_sa.append(np.array([Q[states[i], actions[i]] for i in range(self.n)]))
                vfsp = np.array([0.0 if t[3] is None else np.max(Q[t[3]]) for t in self.data])
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)

        elif q_type == 'keras_standard':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            next_states = np.array([t[3][0] for t in self.data])
            states = np.array([t[0][0] for t in self.data])
            for Q in q_functions:
                qs = Q.predict(states)
                self.q_sa.append(np.array([qs[i][actions[i]] for i in range(self.n)]))
                vfsp = np.max(Q.predict(next_states), axis=1)
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)
































        elif q_type == 'torch_atari':
            batch_size = min(1024, self.data.crt_size, data_size)
            self.data.batch_size = batch_size
            self.q_sa = [np.zeros(data_size) for _ in q_functions]
            self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]
            ptr = 0
            while ptr < data_size:
                state, action, next_state, reward, done = self.data.sample()
                for i, Q in enumerate(q_functions):
                    length = min(batch_size, data_size - ptr)
                    self.q_sa[i][ptr:ptr + length] = Q(state).gather(1, action).cpu().detach().numpy().flatten()[
                                                     :length]
                    vfsp = (reward + Q(next_state) * done * self.gamma).max(dim=1)[0]
                    self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]
                ptr += batch_size
            self.n = data_size

        elif q_type == 'torch_actor_critic_cont':
            batch_size = min(1024, self.data.size, data_size)
            self.data.batch_size = batch_size
            self.q_sa = [np.zeros(data_size) for _ in q_functions]
            self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]
            ptr = 0
            while ptr < data_size:
                length = min(batch_size, data_size - ptr)
                state, action, next_state, reward, done = self.data.sample(length)
                for i, Q in enumerate(q_functions):
                    actor, critic = Q
                    self.q_sa[i][ptr:ptr + length] = critic(state, action).cpu().detach().numpy().flatten()[
                                                     :length]

                    vfsp = (reward + critic(next_state, actor(next_state)) * done * self.gamma)

                    self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]

                ptr += batch_size
            self.n = data_size

        if self.verbose:
            print(F"Data size = {self.n}")
        self.record.avg_q = [np.sum(qsa) for qsa in self.q_sa]
        self.vmax = np.max(self.q_sa)
        self.vmin = np.min(self.q_sa)

    def discretize(self):
        self.q_sa_discrete = []
        self.q_to_data_map = []
        bins = int((self.vmax - self.vmin) / self.res) + 1

        for q in self.q_sa:
            discretized_q = np.digitize(q, np.linspace(self.vmin, self.vmax, bins), right=True)
            self.q_sa_discrete.append(discretized_q)
            q_to_data_map = {}
            for i, q_val in enumerate(discretized_q):
                if q_val not in q_to_data_map:
                    q_to_data_map[q_val] = i
                else:
                    if isinstance(q_to_data_map[q_val], int):
                        q_to_data_map[q_val] = [q_to_data_map[q_val]]
                    q_to_data_map[q_val].append(i)
            self.q_to_data_map.append(q_to_data_map)

    def get_groups(self, q1, q2):
        q1_dic = self.q_to_data_map[q1]
        q2_inds, q2_dic = self.q_sa_discrete[q2], self.q_to_data_map[q2]
        groups = []
        for key in q1_dic:
            if isinstance(q1_dic[key], list):
                q1_list = q1_dic[key]
                set1 = set(q1_list)
                for p1 in q1_list:
                    if p1 in set1 and isinstance(q2_dic[q2_inds[p1]], list):
                        set2 = set(q2_dic[q2_inds[p1]])
                        intersect = set1.intersection(set2)
                        set1 = set1.difference(intersect)
                        if len(intersect) > 1:
                            groups.append(list(intersect))
        return groups

    def compute_loss(self, q1, groups):
        Tf = self.r_plus_vfsp[q1].copy()
        for group in groups:
            Tf[group] = np.mean(Tf[group])
        diff = self.q_sa[q1] - Tf
        return np.sqrt(np.mean(diff ** 2))

    def get_bins(self, groups):
        group_sizes = [len(g) for g in groups]
        bin_ind = np.digitize(group_sizes, self.bins, right=True)
        percent_bins = np.zeros(len(self.bins) + 1)
        count_bins = np.zeros(len(self.bins) + 1)
        for i in range(len(group_sizes)):
            count_bins[bin_ind[i] + 1] += 1
            percent_bins[bin_ind[i] + 1] += group_sizes[i]
        percent_bins[0] = self.n - np.sum(percent_bins)
        count_bins[0] = percent_bins[0]
        return percent_bins, count_bins

    def run(self, resolution=1e-2):
        self.res = resolution
        if self.verbose:
            print(F"Being discretizing outputs of Q functions on batch data with resolution = {resolution}")
        self.discretize()
        if self.verbose:
            print("Starting pairwise comparison")
        percent_histos = []
        count_histos = []
        group_count = []
        loss_matrix = np.zeros((self.q_size, self.q_size))
        for q1 in tqdm(range(self.q_size)):
            for q2 in range(q1, self.q_size):
                groups = self.get_groups(q1, q2)

                group_count.append(len(groups))

                loss_matrix[q1, q2] = self.compute_loss(q1, groups)

                if q1 != q2:
                    loss_matrix[q2, q1] = self.compute_loss(q2, groups)

        average_group_count = np.mean(group_count)
        if self.verbose:
            print(np.max(loss_matrix, axis=1))
        self.record.resolutions.append(resolution)
        self.record.losses.append(np.max(loss_matrix, axis=1))

        self.record.loss_matrices.append(loss_matrix)

        self.record.group_counts.append(average_group_count)

    def compute_optimal_group_skyline(self):
        groups = self.get_groups(self.q_size - 1, self.q_size - 1)
        loss = [self.compute_loss(q, groups) for q in range(self.q_size)]
        self.record.optimal_grouping_skyline.append(np.array(loss))

    def compute_e_q_star_diff(self):
        q_star = self.q_sa[-1]
        e_q_star_diff = [np.sqrt(np.mean((q - q_star) ** 2)) for q in self.q_sa[:-1]] + [0.0]
        self.record.e_q_star_diff = np.array(e_q_star_diff)

    def get_br_ranking(self):
        br = [np.sqrt(np.sum((self.q_sa[q] - self.r_plus_vfsp[q]) ** 2)) for q in range(self.q_size)]
        br_rank = np.argsort(br)
        self.record.bellman_residual = br
        return br_rank

    def get_bvft_ranking(self):

        multi_run_losses = np.vstack(self.record.losses)
        final_loss = np.min(multi_run_losses, axis=0)
        bvft_rank = np.argsort(final_loss)

        return bvft_rank


import conservative_q.discrete_BCQ.bcq_utils as bcq_utils

import torch
from tqdm import tqdm


class BVFT_Sepsis(BVFT):
    def __init__(self, bcq_policies, data, gamma, rmax, rmin, record: BvftRecord = BvftRecord(), q_type='tabular',
                 verbose=False, bins=None, data_size=500):

        self.data = data
        self.gamma = gamma
        self.res = 0
        self.q_sa_discrete = []
        self.q_to_data_map = []
        self.q_size = len(bcq_policies)
        self.verbose = verbose
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100, 1e10]

        self.bins = bins
        self.q_sa = []
        self.r_plus_vfsp = []
        self.q_functions = bcq_policies
        self.bcq_policies = bcq_policies
        self.record = record

    def compute_qsa_r_plus_vfsp(self, valid_df, args, device):

        valid_dataset = bcq_utils.BCDataset(valid_df,
                                            args.feature_names,
                                            args.action_name,
                                            args.reward_name,
                                            args.target_names,
                                            args.trajectory_name,
                                            batch_size=128,
                                            feature_range=args.feature_range,
                                            device=device)

        q_sa = []
        r_plus_vfsp = []

        gamma = 0.99

        for policy in tqdm(self.bcq_policies):
            one_policy_q_sa = []
            one_policy_r_plus_vfsp = []

            end = False
            while not end:
                state, action, next_state, reward, done, pibs, dist_actions, end = valid_dataset.sample()
                with torch.no_grad():
                    q_values, imt, i = policy.Q(state)
                    qa_values = q_values.gather(1, action).reshape(-1, 1)

                    next_state_q, imt, i = policy.Q(next_state)
                    vfsp = (reward + next_state_q * done * gamma).max(dim=1)[0]

                one_policy_q_sa.append(qa_values.cpu())
                one_policy_r_plus_vfsp.append(vfsp.cpu().detach().unsqueeze(1).numpy())

            one_policy_q_sa = np.vstack(one_policy_q_sa)
            q_sa.append(one_policy_q_sa.squeeze())

            one_policy_r_plus_vfsp = np.vstack(one_policy_r_plus_vfsp)
            r_plus_vfsp.append(one_policy_r_plus_vfsp)

        self.q_sa = q_sa
        self.r_plus_vfsp = r_plus_vfsp

        self.record.avg_q = [np.sum(qsa) for qsa in self.q_sa]

        self.vmax = np.max(self.q_sa)
        self.vmin = np.min(self.q_sa)

        self.n = q_sa[0].shape[0]

    def compute_qsa_r_plus_vfsp_fqe(self, valid_df, args, device):
        valid_dataset = bcq_utils.BCDataset(valid_df,
                                            args.feature_names,
                                            args.action_name,
                                            args.reward_name,
                                            args.target_names,
                                            args.trajectory_name,
                                            batch_size=128,
                                            feature_range=args.feature_range,
                                            device=device)

        q_sa = []
        r_plus_vfsp = []

        gamma = 0.99

        for policy in tqdm(self.bcq_policies):
            one_policy_q_sa = []
            one_policy_r_plus_vfsp = []

            end = False
            while not end:
                state, action, next_state, reward, done, pibs, dist_actions, end = valid_dataset.sample()
                with torch.no_grad():
                    q_values = policy.Q(state)
                    qa_values = q_values.gather(1, action).reshape(-1, 1)

                    next_state_q = policy.Q(next_state)
                    vfsp = (reward + next_state_q * done * gamma).max(dim=1)[0]

                one_policy_q_sa.append(qa_values.cpu())
                one_policy_r_plus_vfsp.append(vfsp.cpu().detach().unsqueeze(1).numpy())

            one_policy_q_sa = np.vstack(one_policy_q_sa)
            q_sa.append(one_policy_q_sa.squeeze())

            one_policy_r_plus_vfsp = np.vstack(one_policy_r_plus_vfsp)
            r_plus_vfsp.append(one_policy_r_plus_vfsp)

        self.q_sa = q_sa
        self.r_plus_vfsp = r_plus_vfsp

        self.record.avg_q = [np.sum(qsa) for qsa in
                             self.q_sa]

        self.vmax = np.max(self.q_sa)
        self.vmin = np.min(self.q_sa)

        self.n = q_sa[0].shape[0]


class BVFT_Robomimic(BVFT):
    def __init__(self, num_policies, gamma, rmax, rmin, record: BvftRecord = BvftRecord(), q_type='tabular',
                 verbose=False, bins=None, data_size=500):

        self.gamma = gamma
        self.res = 0
        self.q_sa_discrete = []
        self.q_to_data_map = []
        self.q_size = num_policies
        self.verbose = verbose
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100, 1e10]

        self.bins = bins
        self.q_sa = []
        self.r_plus_vfsp = []

        self.record = record

    def compute_qsa_r_plus_vfsp(self, qvalues):

        lengths = []
        traj_lengths = []

        for estimates in qvalues:
            q_values_all = estimates['qvalues']
            traj_lengths.append(len(q_values_all))
            for q_values in q_values_all:
                lengths.append(q_values.shape[0])
        T = np.min(lengths)
        D_max = np.min(traj_lengths)
        print(f"Shortest common length is {T}")
        print(f"Shortest Number of Trajectory is {D_max}")

        q_sa = []
        r_plus_vfsp = []

        gamma = 0.99

        for estimates in qvalues:
            one_policy_q_sa = []
            one_policy_r_plus_vfsp = []

            q_values_all = estimates['qvalues']
            for d in range(D_max):
                qa_values = q_values_all[d][:T]
                next_state_q = estimates['next_qvalues'][d][:T]

                reward = estimates['rewards'][d][:T]

                vfsp = (reward + next_state_q * gamma).max(dim=1)[0]

                qa_values = qa_values.cpu().detach().numpy()
                vfsp = vfsp.cpu().detach().unsqueeze(1).numpy()

                one_policy_q_sa.append(qa_values)
                one_policy_r_plus_vfsp.append(vfsp)

            one_policy_q_sa = np.vstack(one_policy_q_sa)
            q_sa.append(one_policy_q_sa.squeeze())

            one_policy_r_plus_vfsp = np.vstack(one_policy_r_plus_vfsp)
            r_plus_vfsp.append(one_policy_r_plus_vfsp)

        self.q_sa = q_sa
        self.r_plus_vfsp = r_plus_vfsp

        self.record.avg_q = [np.sum(qsa) for qsa in
                             self.q_sa]

        self.vmax = np.max([np.max(m) for m in self.q_sa])
        self.vmin = np.min([np.min(m) for m in self.q_sa])

        self.n = q_sa[0].shape[0]


from collections import defaultdict


def reassemble_d4rl_tuples(tups, num_traj=100):
    qvalues = tups['qvalues']
    rewards = tups['rewards']
    nextqvalues = tups['nextqvalues']
    dones = tups['dones']

    name_to_stacks_of_traj = defaultdict(list)

    prev_t = 0
    max_t = 999

    curr_iter = 0

    while prev_t < dones.shape[0]:
        t = prev_t + max_t
        if dones[prev_t:t].sum() == 0:
            name_to_stacks_of_traj['dones'].append(dones[prev_t:t])
            name_to_stacks_of_traj['qvalues'].append(qvalues[prev_t:t])
            name_to_stacks_of_traj['next_qvalues'].append(nextqvalues[prev_t:t])
            name_to_stacks_of_traj['rewards'].append(rewards[prev_t:t])
            prev_t = t
        else:

            stopping_points = dones[prev_t:t].nonzero()
            if stopping_points.shape[0] == 1:
                idx = dones[prev_t:t].nonzero().squeeze()[0]
            else:
                idx = stopping_points[0][0]
            name_to_stacks_of_traj['dones'].append(dones[prev_t:prev_t + idx + 1])
            name_to_stacks_of_traj['qvalues'].append(qvalues[prev_t:prev_t + idx + 1])
            name_to_stacks_of_traj['next_qvalues'].append(nextqvalues[prev_t:prev_t + idx + 1])
            name_to_stacks_of_traj['rewards'].append(rewards[prev_t:prev_t + idx + 1])
            prev_t = (prev_t + idx + 1).item()

        curr_iter += 1
        if curr_iter >= num_traj:
            break

    return name_to_stacks_of_traj


class BVFT_D4RL(BVFT):
    def __init__(self, num_policies, gamma, rmax, rmin, record: BvftRecord = BvftRecord(), q_type='tabular',
                 verbose=False, bins=None, data_size=500):

        self.gamma = gamma
        self.res = 0
        self.q_sa_discrete = []
        self.q_to_data_map = []
        self.q_size = num_policies
        self.verbose = verbose
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100, 1e10]

        self.bins = bins
        self.q_sa = []
        self.r_plus_vfsp = []

        self.record = record

    def compute_qsa_r_plus_vfsp(self, qvalues):

        lengths = []
        traj_lengths = []

        for estimates in qvalues:
            q_values_all = estimates['qvalues']
            traj_lengths.append(len(q_values_all))
            for q_values in q_values_all:
                lengths.append(q_values.shape[0])
        T = np.min(lengths)
        D_max = np.min(traj_lengths)
        print(f"Shortest common length is {T}")
        print(f"Shortest Number of Trajectory is {D_max}")

        q_sa = []
        r_plus_vfsp = []

        gamma = 0.99

        for estimates in qvalues:
            one_policy_q_sa = []
            one_policy_r_plus_vfsp = []

            q_values_all = estimates['qvalues']
            for d in range(D_max):
                qa_values = q_values_all[d][:T]
                next_state_q = estimates['next_qvalues'][d][:T]

                reward = estimates['rewards'][d][:T]
                dones = estimates['dones'][d][:T]

                vfsp = (reward + (1 - dones) * next_state_q * gamma).max(dim=1)[0]

                qa_values = qa_values.cpu().detach().numpy().reshape(-1)
                vfsp = vfsp.cpu().detach().view(-1).unsqueeze(1).numpy()

                one_policy_q_sa.append(qa_values)
                one_policy_r_plus_vfsp.append(vfsp)

            one_policy_q_sa = np.vstack(one_policy_q_sa).reshape(-1)
            q_sa.append(one_policy_q_sa.squeeze())

            one_policy_r_plus_vfsp = np.vstack(one_policy_r_plus_vfsp)
            r_plus_vfsp.append(one_policy_r_plus_vfsp)

        self.q_sa = q_sa
        self.r_plus_vfsp = r_plus_vfsp

        self.record.avg_q = [np.sum(qsa) for qsa in
                             self.q_sa]

        self.vmax = np.max([np.max(m) for m in self.q_sa])
        self.vmin = np.min([np.min(m) for m in self.q_sa])

        self.n = q_sa[0].shape[0]


if __name__ == '__main__':
    test_data = [(0, 0, 1.0, 1), (0, 1, 1.0, 2),
                 (1, 0, 0.0, None), (1, 1, 0.0, None),
                 (2, 0, 1.0, None), (2, 1, 1.0, None),
                 (3, 0, 1.0, 4), (3, 1, 1.0, 4)]

    Q1 = np.array([[1.0, 1.9], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    Q2 = np.array([[7.0, 1.9], [0.0, 0.0], [1.0, 1.0], [7.0, 7.0], [10.0, 10.0]])

    gamma = 0.9
    rmax, rmin = 1.0, 0.0
    record = BvftRecord()
    b = BVFT([Q1, Q2], test_data, gamma, rmax, rmin, record, q_type='tabular', verbose=True)
    b.run()

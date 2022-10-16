import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Conv_Q(nn.Module):
    def __init__(self, frames, num_actions):
        super(Conv_Q, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(3136, 512)
        self.l2 = nn.Linear(512, num_actions)

    def forward(self, state):
        q = F.relu(self.c1(state))
        q = F.relu(self.c2(q))
        q = F.relu(self.c3(q))
        q = F.relu(self.l1(q.reshape(-1, 3136)))
        return self.l2(q)


class FC_Q(nn.Module):
    def __init__(self, state_dim, hid_dim, num_actions):
        super(FC_Q, self).__init__()
        self.l1 = nn.Linear(state_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, num_actions)

    def forward(self, state):
        q = F.gelu(self.l1(state))
        q = F.gelu(self.l2(q))
        return self.l3(q)


class discrete_FQE(object):
    def __init__(
            self,
            is_atari,
            num_actions,
            state_dim,
            device,
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
            hid_dim=128,
    ):

        self.device = device

        self.Q = Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, hid_dim,
                                                                                         num_actions).to(
            self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
        self.num_actions = num_actions

        self.iterations = 0

    def train(self, replay_buffer):

        state, action, next_state, reward, not_done, end = replay_buffer.sample()

        with torch.no_grad():
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

        current_Q = self.Q(state).gather(1, action)

        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        self.iterations += 1
        self.maybe_update_target()

        return Q_loss.cpu().detach().item(), end

    def full_train(self, replay_buffer, epochs=10):

        q_losses = []
        replay_buffer.shuffle()
        for _ in tqdm(range(epochs)):
            end = False
            while not end:
                q_loss, end = self.train(replay_buffer)
                q_losses.append(q_loss)
        return q_losses

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)


def create_fqe_for_sepsis(bc_dataset, action_size=8, device=None, lr=3e-4, target_update_frequency=10,
                          hid_dim=128):
    fqe = discrete_FQE(is_atari=False, num_actions=action_size, hid_dim=hid_dim,
                       state_dim=len(bc_dataset.feature_names),
                       device=device, optimizer_parameters={"lr": lr},
                       target_update_frequency=target_update_frequency,
                       polyak_target_update=True)
    return fqe


def get_q_values_per_traj(valid_df, bc_dataset, fqe, evaluation_policy, device=None):
    initial_states = valid_df.groupby("Trajectory").first()
    initial_states = np.asarray(initial_states[bc_dataset.feature_names]).astype(float)
    initial_states = torch.FloatTensor(initial_states)

    with torch.no_grad():
        prob = evaluation_policy.softmax_prob(initial_states)
        agent_action = np.argmax(prob, axis=1)

    agent_action = torch.LongTensor(agent_action).unsqueeze(1).to(device)
    initial_states = initial_states.to(device)

    with torch.no_grad():
        Qs = fqe.Q(initial_states).gather(1, agent_action)

    return Qs


class StandardBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.pibs = np.zeros((self.max_size, 25))

        self.nndist_action = np.zeros(
            (self.max_size, 25))

    def add(self, state, action, next_state, reward, done, episode_done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")


class ExhaustiveBuffer(StandardBuffer):
    def __init__(self, state_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.pibs = np.zeros((self.max_size, 25))

        self.nndist_action = np.zeros(
            (self.max_size, 25))

        self.index = 0

    def add(self, state, action, next_state, reward, done, episode_done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        states = torch.FloatTensor(self.state[self.index:self.index + batch_size]).to(self.device)
        actions = torch.LongTensor(self.action[self.index:self.index + batch_size]).to(self.device)
        rewards = torch.FloatTensor(self.reward[self.index:self.index + batch_size]).to(self.device)
        nexts = torch.FloatTensor(self.next_state[self.index:self.index + batch_size]).to(self.device)
        done = torch.BoolTensor(self.not_done[self.index:self.index + batch_size]).to(self.device)
        end = False
        self.index += batch_size
        if self.index >= self.state.shape[0]:
            self.shuffle()
            end = True

        return states, actions, nexts, rewards, done, end

    def shuffle(self):
        indices = np.arange(self.state.shape[0])
        np.random.shuffle(indices)

        self.state = self.state[indices]
        self.action = self.action[indices]
        self.next_state = self.next_state[indices]
        self.reward = self.reward[indices]
        self.not_done = self.not_done[indices]
        self.index = 0


def generate_replay_buffer(bc_dataset, evaluation_policy, device=None):
    replay_buffer = ExhaustiveBuffer(state_dim=len(bc_dataset.feature_names),
                                     batch_size=bc_dataset.batch_size,
                                     buffer_size=len(bc_dataset),
                                     device=device)
    end = False
    while not end:
        state, action, next_state, reward, done, pibs, dist_actions, end = bc_dataset.sample()

        state = state.cpu()
        next_state = next_state.cpu()
        reward = reward.cpu()
        done = done.cpu().float()
        with torch.no_grad():
            prob = evaluation_policy.softmax_prob(state)

            agent_action = np.argmax(prob, axis=1)
            for idx in range(state.shape[0]):
                replay_buffer.add(state[idx], agent_action[idx], next_state[idx], reward[idx], done[idx], None, None)

    return replay_buffer


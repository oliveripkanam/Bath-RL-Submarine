import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .model import DQN, DuelingDQN

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DoubleDQNAgent:
    def __init__(self, input_shape=19, num_actions=4, lr=1e-5, gamma=0.99, buffer_size=500000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device("cpu")

        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        
        # Sync target network with policy network initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(batch_size)

        state_batch = torch.FloatTensor(np.array(state)).to(self.device)
        action_batch = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state)).to(self.device)
        done_batch = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Compute q-values for current state
        q_values = self.policy_net(state_batch)
        q_value = q_values.gather(1, action_batch)

        # Compute target q-values using double dqn
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            expected_q_value = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_value, expected_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DuelingDQNAgent(DoubleDQNAgent):
    def __init__(self, input_shape=19, num_actions=4, lr=1e-5, gamma=0.99, buffer_size=100000):
        super().__init__(input_shape, num_actions, lr, gamma, buffer_size)

        # Override the networks with DuelingDQN
        self.policy_net = DuelingDQN(input_shape, num_actions).to(self.device)
        self.target_net = DuelingDQN(input_shape, num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Re-init optimizer because parameters changed
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from .model import DQN
from .per import PrioritizedReplayBuffer

class DoubleDQNAgent:
    def __init__(self, input_shape=19, num_actions=4, lr=1e-5, gamma=0.99, buffer_size=100000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device("cpu")

        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        
        # Sync target network with policy network initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def train_step(self, batch_size, beta=0.4):
        if len(self.memory) < batch_size:
            return

        state, action, reward, next_state, done, idxs, weights = self.memory.sample(batch_size, beta)

        state_batch = torch.FloatTensor(np.array(state)).to(self.device)
        action_batch = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state)).to(self.device)
        done_batch = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        weights_batch = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Compute q-values for current state
        q_values = self.policy_net(state_batch)
        q_value = q_values.gather(1, action_batch)

        # Compute target q-values using double dqn
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            expected_q_value = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Calculate element-wise loss for PER update
        loss_elementwise = F.smooth_l1_loss(q_value, expected_q_value, reduction='none')
        
        # Weighted loss for optimization
        loss = (loss_elementwise * weights_batch).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        errors = torch.abs(q_value - expected_q_value).detach().cpu().numpy()
        self.memory.update_priorities(idxs, errors)

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

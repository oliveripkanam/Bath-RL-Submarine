import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .model import PPOActorCritic

class PPOAgent:
    def __init__(self, input_shape=39, num_actions=5, lr=3e-4, gamma=0.99, clip_ratio=0.2,
                 value_coeff=0.5, entropy_coeff=0.01, max_grad_norm=0.5, epochs=10, batch_size=64):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = PPOActorCritic(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(state_t)
        return action, log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value=0, lam=0.95):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        return np.array(returns), np.array(advantages)

    def train_step(self):
        if len(self.states) == 0:
            return None

        with torch.no_grad():
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            _, _, last_value = self.model.get_action(last_state)
            last_value = last_value.item()

        returns, advantages = self.compute_gae(last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        t_actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        t_old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        t_returns = torch.FloatTensor(returns).to(self.device)
        t_advantages = torch.FloatTensor(advantages).to(self.device)

        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        avg_loss = 0
        n_updates = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                mb_states = t_states[idx]
                mb_actions = t_actions[idx]
                mb_old_log_probs = t_old_log_probs[idx]
                mb_returns = t_returns[idx]
                mb_advantages = t_advantages[idx]

                new_log_probs, values, entropy = self.model.evaluate_actions(mb_states, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                avg_loss += loss.item()
                n_updates += 1

        self.clear_memory()
        
        return avg_loss / n_updates if n_updates > 0 else 0

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
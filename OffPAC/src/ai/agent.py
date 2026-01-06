import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .model import ActorCriticNetwork

class OffPACReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mu):
        # mu is the probability of the taken action under the behavior policy at the time
        self.buffer.append((state, action, reward, next_state, done, mu))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mu = zip(*batch)
        return state, action, reward, next_state, done, mu

    def __len__(self):
        return len(self.buffer)

class OffPACAgent:
    def __init__(self, input_shape=39, num_actions=5, lr=1e-4, gamma=0.99, buffer_size=500000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network setup for actor-critic
        self.model = ActorCriticNetwork(input_shape, num_actions).to(self.device)
        self.target_model = ActorCriticNetwork(input_shape, num_actions).to(self.device)
        
        # Sync target network initially
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Shared optimizer for both heads
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = OffPACReplayBuffer(buffer_size)
        self.batch_size = 128

    def select_action(self, state, epsilon=0.0):
        """
        Select action using the current policy mixed with epsilon noise (exploration).
        Returns: 
            action (int): The chosen action.
            mu (float): The probability of selecting this action (for Retrace).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_probs, _ = self.model(state_t)
            policy_probs = policy_probs.cpu().numpy()[0]

        # Select action using epsilon-greedy mixture of random and learned policy
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.random.choice(self.num_actions, p=policy_probs)
            
        # Calculate behavior probability (mu) as mixture of uniform and policy probabilities
        mu = (epsilon / self.num_actions) + ((1 - epsilon) * policy_probs[action])
        
        return action, mu

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        state, action, reward, next_state, done, mu = self.memory.sample(batch_size)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        mu = torch.FloatTensor(mu).unsqueeze(1).to(self.device)

        # Get current policy and Q-values from the model
        pi, q_values = self.model(state)
        q_value_taken = q_values.gather(1, action)
        
        # Compute target values using target network
        with torch.no_grad():
            next_pi, next_q_values = self.target_model(next_state)
            next_v = (next_pi * next_q_values).sum(1, keepdim=True)
            
        # Calculate importance weights rho
        pi_picked = pi.gather(1, action)
        rho = pi_picked / (mu + 1e-6)
        
        # Truncate importance weights for stability
        rho_bar = torch.clamp(rho, max=1.0)
        
        # Compute Q-target
        q_target = reward + self.gamma * next_v * (1 - done)
        
        # Calculate critic loss
        critic_loss = F.mse_loss(q_value_taken, q_target)
        
        # Calculate advantage
        v_curr = (pi * q_values.detach()).sum(1, keepdim=True)
        advantage = q_value_taken.detach() - v_curr
        
        # Calculate log probabilities
        log_pi = torch.log(pi + 1e-10)
        log_pi_picked = log_pi.gather(1, action)
        
        # Calculate actor loss
        actor_loss = -(rho_bar.detach() * advantage * log_pi_picked).mean()
        
        # Add entropy regularization
        entropy = -(pi * log_pi).sum(1).mean()
        
        # Combine losses
        total_loss = critic_loss + actor_loss - 1e-3 * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())

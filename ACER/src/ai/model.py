import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape=39, num_actions=4):
        """
        Actor-Critic Network for ACER.
        
        Args:
            input_shape (int): Size of the state vector.
            num_actions (int): Number of discrete actions.
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature layers
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # Actor head
        self.actor = nn.Linear(256, num_actions)
        
        # Critic head
        self.critic = nn.Linear(256, num_actions)
        
    def forward(self, x):
        """
        Forward pass.
        Returns:
            policy_probs: Softmax probabilities for actions.
            q_values: Q-values for all actions.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Softmax over logits
        policy_logits = self.actor(x)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Linear Q-values
        q_values = self.critic(x)
        
        return policy_probs, q_values

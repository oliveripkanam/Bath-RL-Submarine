import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOActorCritic(nn.Module):
    def __init__(self, input_shape=39, num_actions=5):
        super(PPOActorCritic, self).__init__()
        
        self.fc1 = init_layer(nn.Linear(input_shape, 512))
        self.fc2 = init_layer(nn.Linear(512, 512))
        self.fc3 = init_layer(nn.Linear(512, 256))
        
        self.actor_fc = init_layer(nn.Linear(256, 128))
        self.actor_out = init_layer(nn.Linear(128, num_actions), std=0.01)
        
        self.critic_fc = init_layer(nn.Linear(256, 128))
        self.critic_out = init_layer(nn.Linear(128, 1), std=1.0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def get_action(self, state):
        features = self.forward(state)
        
        actor_features = F.relu(self.actor_fc(features))
        logits = self.actor_out(actor_features)
        dist = torch.distributions.Categorical(logits=logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        critic_features = F.relu(self.critic_fc(features))
        value = self.critic_out(critic_features)
        
        return action.item(), log_prob, value

    def evaluate_actions(self, states, actions):
        features = self.forward(states)
        
        actor_features = F.relu(self.actor_fc(features))
        logits = self.actor_out(actor_features)
        dist = torch.distributions.Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        critic_features = F.relu(self.critic_fc(features))
        values = self.critic_out(critic_features)
        
        return log_probs, values.squeeze(-1), entropy
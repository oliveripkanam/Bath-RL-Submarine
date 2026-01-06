import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape=19, num_actions=4):
        """
        Deep Q-Network (DQN) for Submarine Navigation.
        
        Args:
            input_shape (int): Size of the state vector (16 Sonar + 1 Battery + 2 Velocity = 19).
            num_actions (int): Number of discrete actions (4: Up, Down, Left, Right).
        """
        super(DQN, self).__init__()
        
        # Neural network layers
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_actions)
        
    def forward(self, x):
        """
        Forward pass through the network.
        Input: State Tensor
        Output: Q-Values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape=19, num_actions=4):
        """
        Dueling DQN Architecture
        Splits Q-value estimation into Value (V) and Advantage (A) streams.
        """
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 512)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals
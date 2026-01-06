import matplotlib.pyplot as plt
import numpy as np

agent_prefixes = ['ddqn', 'vanilla_dqn', 'ppo']
agent_dict = {}

for prefix in agent_prefixes:
    try:
        loss_data = np.load(f'models/{prefix}_training_loss.npy')
        agent_dict[prefix.upper()] = loss_data
    except FileNotFoundError:
        print(f"Warning: No loss data found for {prefix.upper()} agent.")
        
plt.figure(figsize=(10, 6))
for agent_name, loss_data in agent_dict.items():
    plt.plot(loss_data, label=agent_name)
plt.xlabel('Episode')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Episodes for Different Agents')
plt.legend()
plt.grid(True)
plt.show()

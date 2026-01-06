## Submarine Navigation with Reinforcement Learning
This project implements and compares six Deep Reinforcement Learning (DRL) architectures trained to pilot a submarine through complex 2D cave systems. The agents must manage continuous momentum physics, avoid obstacles, and collect resources (batteries) to reach a target destination.

## Implemented Agents
DQN
Double DQN
Dueling DQN
PER
PPO
Off-PAC

## Setup
pip install pymunk pygame numpy torch

## How to Run
**Manual Play:**
`python main.py`

**Train AI Agent:**
`python train.py`
- Press **TAB** to toggle Fast Mode (Headless) vs Watch Mode.

## Controls (Manual Mode)
- **Arrow Keys:** Move Submarine
- **1-8:** Change Map Selection
- **Shift Key:** Pitch Black Mode

## Integration Guide
The Sonar module is self-contained in `src/sonar/sensors.py`.

```python
from src.sonar.sensors import Sonar

# Init (Before loop)
my_sonar = Sonar(space, sub_body)

# In Game Loop
state = my_sonar.get_observation() # Returns 16 normalized floats (0.0-1.0)
my_sonar.draw(screen, font)        # Visual debug lines
```

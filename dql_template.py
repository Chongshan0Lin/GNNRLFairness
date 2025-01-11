import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Set random seeds for reproducibility (optional but good practice)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------
# 1. Hyperparameters
# -------------------------------
ENV_NAME = "CartPole-v1"
BATCH_SIZE = 64
GAMMA = 0.99           # Discount factor
LR = 1e-3              # Learning rate
MEMORY_SIZE = 100000   # Replay buffer size
MIN_MEMORY_FOR_TRAIN = 1000
EPS_START = 1.0        # Exploration rate (epsilon) at start
EPS_END = 0.01         # Minimum epsilon
EPS_DECAY = 500        # Decay factor for epsilon
TARGET_UPDATE_FREQ = 10
MAX_EPISODES = 500

# -------------------------------
# 2. Create the environment
# -------------------------------
env = gym.make(ENV_NAME)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# -------------------------------
# 3. Define the Q-Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# 4. Initialize policy & target networks
# -------------------------------
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # Copy weights initially
target_net.eval()  # Target net in inference mode by default

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# -------------------------------
# 5. Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

memory = ReplayBuffer(MEMORY_SIZE)

# -------------------------------
# 6. Epsilon-greedy action selection
# -------------------------------
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_t)
            return q_values.argmax(dim=1).item()

# -------------------------------
# 7. Training function for one batch
# -------------------------------
def train_step():
    # Don't train until we have enough experience in the replay buffer
    if len(memory) < MIN_MEMORY_FOR_TRAIN:
        return
    
    # Sample a batch
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions).unsqueeze(1)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.FloatTensor(dones).unsqueeze(1)

    # Current Q values (policy network)
    q_values = policy_net(states_t).gather(1, actions_t)

    # Next Q values (target network)
    with torch.no_grad():
        max_next_q_values = target_net(next_states_t).max(dim=1, keepdim=True)[0]

    # Q target = reward + (gamma * max_next_q_value * (1 - done))
    q_targets = rewards_t + (GAMMA * max_next_q_values * (1 - dones_t))

    # Compute loss
    loss = loss_fn(q_values, q_targets)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -------------------------------
# 8. Main Training Loop
# -------------------------------
epsilon = EPS_START
all_rewards = []

for episode in range(MAX_EPISODES):
    state = env.reset()
    episode_reward = 0

    done = False
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        # Store transition in replay buffer
        memory.push(state, action, reward, next_state, done)
        
        # Update state
        state = next_state
        episode_reward += reward
        
        # Training step
        train_step()
        
        # Decay epsilon
        epsilon = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-1.0 * episode / EPS_DECAY)

    all_rewards.append(episode_reward)

    # Update the target network periodically
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(all_rewards[-10:])
        print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

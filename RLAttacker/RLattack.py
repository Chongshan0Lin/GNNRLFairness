import torch.nn as nn
import networkx as nx
import torch
import gym
import random
import numpy as np
from graph_embedding import s2v_embedding
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_size = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.fc3(x)

class replay_buffer:

    """
    Replay buffer data structure for agent to 
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return(np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32))

    def length(self):
        return len(self.buffer)
    

    

class agent:
    def __init__(self, graph, feature_matrix, labels, state_dim, idx_test, idx_train, epsilon, min_memory_step, buffer_capacity = 200000):
        """
        graph: networkx graph
        feature_matrix: tensor matrix that store the features of nodes
        labels: labels of nodes
        state_dim: size of state space, which is equal to size of graph embedding
        idx_test: indexes of testing set
        idx_train: index of training set
        epsilon: exploration rate
        min_memory_step: minimal number of steps stored in memory when memory replay is allowed to happen
        buffer_capacity: size of replay buffer
        """
        # super().__init__()
        self.graph = graph
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.idx_test = idx_test
        self.idx_train = idx_train

        self.replay_buffer = replay_buffer(capacity=buffer_capacity)

        # This serves as the output dimension of q deep network
        self.action_size = graph.number_of_nodes()
        # This serves as the input dimension of q deep network
        self.state_size = state_dim
        # Exploration rate
        self.epsilon = epsilon
        self.min_memory_step = min_memory_step
        self.embedding = s2v_embedding(graph=self.graph, feature_matrix=self.feature_matrix, output_dim=self.state_size)

        # Create deeq q network
        self.build_Q_network()
        
    def build_Q_network(self):
        """
        Initialize the two networks: policy network and target network
        """
        self.policy_network = DQN(self.state_size, self.action_size)
        self.target_network = DQN(self.state_size, self.action_size)
    
    def select_action(self):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():

                state_t = self.embedding
                q_values = self.policy_network(state_t)
                return q_values.argmax(dim = 1).item()
    
    def train_setp(self):
        if len(self.replay_buffer.length()) < self.min_memory_step:
            return
        
        BATCH_SIZE = 100
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q values (policy network)
        q_values = self.policy_network(states_t).gather(1, actions_t)

        # Next Q values (target network)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]

        # Q target = reward + (gamma * max_next_q_value * (1 - done))
        q_targets = rewards_t + (max_next_q_values * (1 - dones_t))

        loss = loss_fn(q_values, q_targets)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

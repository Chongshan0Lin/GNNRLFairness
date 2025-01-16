import torch.nn as nn
import networkx as nx
import torch
import gym
import random
import numpy as np
from .graph_embedding import s2v_embedding
from collections import deque
# from ..GCN.model import GCN
from GCN.victim import victim
import torch.optim as optim

ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME)
loss_fn = nn.MSELoss()



class DQN(nn.Module):
    """
    The basic structure of one deep q network
    Argument: state_size, action_size, hidden_layer_size
    It has three layers, one state_size * hidden_layer_size, one hidden_layer_size * hidden_layer_size, one hidden_layer_size * action_size
    The forwarding uses relu as non-linear layer
    """
    def __init__(self, state_size, action_size, hidden_layer_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.fc3(x)

class Q_function:
    """
    A Q function that serves as a component the hierarchical Q network
    Argument: order, state_size, action_size, hidden_layer_size (optional), buffer_capacity
    Components: policy network, target network, order, replay_buffer
    """
    def __init__(self, order, state_size, action_size, buffer_capacity, exploration_rate, hidden_layer_size = 128):
        self.order = order
        self.policy_network = DQN(state_size, action_size, hidden_layer_size)
        self.target_network = DQN(state_size, action_size, hidden_layer_size)
        self.replay_buffer = replay_buffer(capacity=buffer_capacity)
        self.state_size = state_size
        self.action_size = action_size
        self.build_Q_network()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3, weight_decay=0)
        self.exploration_rate = exploration_rate


    def build_Q_network(self):
        """
        Initialize the two networks: policy network and target network
        """
        self.policy_network = DQN(self.state_size, self.action_size, hidden_layer_size=(self.state_size * 2) // 3 + self.action_size)
        self.target_network = DQN(self.state_size, self.action_size, hidden_layer_size=(self.state_size * 2) // 3 + self.action_size)



    def select_action(self, state_t):
        """
        Based on the calculation of network, select the proper Q network
        """
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.policy_network.forward(state_t)
                return q_values.argmax(dim = 1).item()

    def train_step(self):
        if len(self.replay_buffer.length()) < self.min_memory_step:
            return

        BATCH_SIZE = 100
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        
        q_values = self.policy_network.forward(states_t).gather(1, actions_t)

        # Next Q values (target network)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]

        # Q target = reward + (gamma * max_next_q_value * (1 - done))
        q_targets = rewards_t + (max_next_q_values * (1 - dones_t))
        loss = loss_fn(q_values, q_targets)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())



class replay_buffer:

    """
    Replay buffer data structure for agent to play memory buffer
    Argument: capacity: max size of replay buffer
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return(np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32))

    def length(self):
        return len(self.buffer)


class agent:
    def __init__(self, graph, feature_matrix, labels, state_dim, idx_test, idx_train, epsilon, min_memory_step, budget, buffer_capacity = 200000):
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
        self.nnodes = graph.number_of_nodes()
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.idx_test = idx_test
        self.idx_train = idx_train
        self.state_dim = state_dim

        action_size = graph.number_of_nodes()

        self.Q_function1 = Q_function(state_size=state_dim, action_size=action_size, order=1, buffer_capacity=buffer_capacity, exploration_rate=epsilon)
        self.Q_function2 = Q_function(state_size=state_dim, action_size=action_size, order=2, buffer_capacity=buffer_capacity, exploration_rate=epsilon)

        self.min_memory_step = min_memory_step
        self.budegt = budget

        self.embedding = s2v_embedding(nnodes=self.nnodes, feature_matrix=self.feature_matrix, output_dim=self.state_dim)

    def change_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)
        else:
            self.graph.add_edge(node1, node2)


    def train(self):
        """
        Train the two agent hierarchically.
        First Q function gives the first node, second Q function gives the second node.
        The reward is based on the difference of fairness after the node is removed.
        The agent runs until it runs out of the budget or successfully achieves the goal
        """

        n_episodes = 20
        print("Number of episodes", n_episodes)
        for episode in range(n_episodes):

            all_rewards = []
            cumulative_reward = 0

            # Create a victim model and train
            victim_model = victim()
            victim_model.train()
            fairness_loss = victim_model.evaluate()
            emb_matrix = self.embedding.n2v(self.graph)
            state_embedding = self.embedding.g2v(emb_matrix)
            # Launch a cycle of attac
            for i in range(self.budegt):

                print(i,"th iteration")
                # Get the current state embedding
                # Select the first node:
                # How shall I make sure that the first node is different from the second node?
                first_node = self.Q_function1.select_action(state_embedding)
                second_node = self.Q_function1.select_action(state_embedding)

                # victim_model.change_edge(first_node, second_node)
                self.change_edge(first_node, second_node)
                victim_model.update_adj_matrix(torch.from_numpy(nx.to_numpy_array(self.graph)))

                # After changing the model, retrain the victim model and calculate the new fairness value
                victim_model.train()
                new_loss = victim_model.evaluate()
                # Determine the difference of fairness, which is the reward
                reward = new_loss - fairness_loss
                cumulative_reward += reward
                fairness_loss = new_loss

                # Update the embedding correspondingly as the new state
                emb_matrix = self.embedding.n2v(self.graph)
                new_state_embedding = self.embedding.g2v(emb_matrix)
                self.Q_function1.replay_buffer.push(state=state_embedding, action=first_node, reward=reward,next_state=new_state_embedding, done=False)
                self.Q_function1.replay_buffer.push(state=state_embedding, action=second_node, reward=reward,next_state=new_state_embedding, done=False)
                state_embedding = new_state_embedding

            all_rewards.append(cumulative_reward)
            # Update the target network periodically
            if episode % 2 == 0:
                self.Q_function1.target_network.load_state_dict(self.Q_function1.policy_network_network.state_dict())
                self.Q_function2.target_network.load_state_dict(self.Q_function2.policy_network_network.state_dict())

            print(episode, "th episode")
            if (episode + 1) % 2 == 0:
                avg_reward = np.mean(all_rewards[-10:])
                print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}")

    def train_step(self):
        """
        Train Q network periodically and minimize loss
        """
        self.Q_function1.train_step()
        self.Q_function2.train_step()
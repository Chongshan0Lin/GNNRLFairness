import torch.nn as nn
import networkx as nx
import torch
import random
import numpy as np
from .graph_embedding import s2v_embedding
from collections import deque
from GCN.victim import victim
import torch.optim as optim
from database import MetricsLogger
from .kernel_estimator import KernelEstimator
# import tensor
gpu_index = 2

loss_fn = nn.MSELoss()



class DQN(nn.Module):
    """
    The basic structure of one deep q network
    Argument: state_size, action_size, hidden_layer_size
    It has three layers, one state_size * hidden_layer_size, one hidden_layer_size * hidden_layer_size, one hidden_layer_size * action_size
    The forwarding uses relu as non-linear layer
    """
    def __init__(self, state_size, action_size, hidden_layer_size, dropout_rate=0.5):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(hidden_layer_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class Q_function:
    """
    A Q function that serves as a component the hierarchical Q network
    Argument: order, state_size, action_size, hidden_layer_size (optional), buffer_capacity
    Components: policy network, target network, order, replay_buffer
    """
    def __init__(self, order, state_size, action_size, buffer_capacity, exploration_rate, min_memory_step, hidden_layer_size = 128, dropout_rate=0.5):
        self.order = order
        self.policy_network = DQN(state_size, action_size, hidden_layer_size, dropout_rate=dropout_rate)
        self.target_network = DQN(state_size, action_size, hidden_layer_size, dropout_rate=dropout_rate)
        self.replay_buffer = replay_buffer(capacity=buffer_capacity)
        self.min_memory_step = min_memory_step
        self.state_size = state_size
        self.action_size = action_size
        self.build_Q_network()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-4, weight_decay=1e-5)
        self.exploration_rate = exploration_rate
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        self.policy_network.to(device)
        self.target_network.to(device)
        print(device)



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
                max_node = torch.argmax(q_values).item()
                return max_node

    def train_step(self):
        if self.replay_buffer.length() < self.min_memory_step:
            print("Current memory size:",self.replay_buffer.length())
            self.exploration_rate = 1.0
            return


        # Here, we are sampling from the memory buffer and perform minibatch gradient descent.
        BATCH_SIZE = self.min_memory_step // 10
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(1)

        # states_t = torch.FloatTensor(states)
        # actions_t = torch.LongTensor(actions).unsqueeze(1)
        # rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        # next_states_t = torch.FloatTensor(next_states)
        # dones_t = torch.FloatTensor(dones).unsqueeze(1)
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")

        # states_t = torch.FloatTensor(torch.stack(states))
        # actions_t = torch.LongTensor(torch.stack(actions)).unsqueeze(1)
        # rewards_t = torch.FloatTensor(torch.stack(rewards)).unsqueeze(1)
        # next_states_t = torch.FloatTensor(torch.stack(next_states))
        # dones_t = torch.LongTensor(torch.stack(dones)).unsqueeze(1)
        states_t = torch.stack(states).to(device).float()
        actions_t = torch.stack(actions).to(device).long().unsqueeze(1)
        rewards_t = torch.stack(rewards).to(device).float().unsqueeze(1)
        next_states_t = torch.stack(next_states).to(device).float()
        dones_t = torch.stack(dones).to(device).long().unsqueeze(1)


        # print(f"states_t shape: {states_t.shape}")          # Should be [BATCH_SIZE, state_dim]
        # print(f"actions_t shape: {actions_t.shape}")        # Should be [BATCH_SIZE, 1]
        # print(f"rewards_t shape: {rewards_t.shape}")        # Should be [BATCH_SIZE, 1]
        # print(f"next_states_t shape: {next_states_t.shape}")# Should be [BATCH_SIZE, state_dim]
        # print(f"dones_t shape: {dones_t.shape}")            # Should be [BATCH_SIZE, 1]

        # approximated_q_values = self.policy_network(states_t)
        # print(approximated_q_values)
        # print(actions_t)
        # q_values = self.policy_network.forward(states_t).gather(1, actions_t)
        
        q_values = self.policy_network(states_t).gather(1, actions_t)  # [BATCH_SIZE, 1]


        # Next Q values (target network)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]

            q_targets = rewards_t + max_next_q_values
        # Q target = reward + (gamma * max_next_q_value * (1 - done))
        # q_targets = rewards_t + (max_next_q_values * (1 - dones_t))
        # print("Doesn_t", dones_t)
        # print("Max next q values:", max_next_q_values)
        # identity = torch.diagonal(torch.eye(dones_t.shape[0]))
        # dones_t = torch.sub(identity, dones_t)
        # mnqv = torch.diagonal(torch.mul(max_next_q_values, dones_t))
        # # q_targets = torch.add(input=rewards_t, other = max_next_q_values, alpha=(1 - dones_t))
        # q_targets = max_next_q_values

        # Make sure the rewards_t has the correct dimension
        # print("Q values and Q targets:")
        # print(q_values)
        # print(q_targets)
        # print("Q values:", q_values)
        # print("Q targets:", q_targets)
        loss = loss_fn(q_values, q_targets)

        # Backdrop
        self.optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()

        # Check if gradient is flowing
        # print("named parameters: ",self.policy_network.named_parameters())
        # for name, param in self.policy_network.named_parameters():
        #     # print("param.grad:",param.grad)
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.abs().sum()}")
        # for name, param in self.embedding.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for embedding {name}: {param.grad.abs().sum()}")

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
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")

        done = int(done)
        # self.buffer.append((state, torch.tensor(action), torch.tensor([reward]), next_state, torch.tensor([done])))
        self.buffer.append((
            state.detach(),  
            torch.tensor(action, device=device, dtype=torch.long),
            torch.tensor([reward], device=device, dtype=torch.float),
            next_state.detach(),  
            torch.tensor([done], device=device, dtype=torch.long)
        ))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # return(torch.from_numpy(np.array(state)),
        #         torch.from_numpy(np.array(action)),
        #         torch.from_numpy(np.array(reward, dtype=np.float32)),
        #         torch.from_numpy(np.array(next_state), np.array(done, dtype=np.float32))
        #         )

        # return(torch.flatten((states)),
        #         torch.flatten((actions)),
        #         torch.flatten((rewards)),
        #         torch.flatten(next_states), 
        #         np.array(dones, dtype=np.float32)
        # )
        return (states, actions, rewards, next_states, dones)

        # return (np.array(states),
        #         np.array(actions),
        #         np.array(rewards, dtype=np.float32),
        #         np.array(next_states),
        #         np.array(dones, dtype=np.float32))

    def length(self):
        return len(self.buffer)


class agent:
    def __init__(self, graph, feature_matrix, labels, state_dim, idx_test, idx_train, epsilon, min_memory_step, budget, buffer_capacity = 20000000):
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

        self.Q_function1 = Q_function(state_size=state_dim, action_size=action_size, order=1, buffer_capacity=buffer_capacity, min_memory_step = min_memory_step, exploration_rate=epsilon)
        self.Q_function2 = Q_function(state_size=state_dim, action_size=action_size, order=2, buffer_capacity=buffer_capacity, min_memory_step = min_memory_step, exploration_rate=epsilon)

        self.min_memory_step = min_memory_step
        self.budegt = budget

        self.embedding = s2v_embedding(nnodes=self.nnodes, feature_matrix=self.feature_matrix, output_dim=self.state_dim)
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        self.embedding.to(device)

        self.metrics_logger = MetricsLogger(db_name="training_metrics.db")

        # self.optimizer = optim.Adam(
        #     list(self.Q_function1.policy_network.parameters()) +
        #     list(self.Q_function2.policy_network.parameters()) +
        #     list(self.embedding.parameters()), 
        #     lr=1e-3,
        #     weight_decay=0
        # )


    def change_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)
        else:
            self.graph.add_edge(node1, node2)


    def train(self, n_episodes, alpha = 1):
        """
        Train the two agent hierarchically.
        First Q function gives the first node, second Q function gives the second node.
        The reward is based on the difference of fairness after the node is removed.
        The agent runs until it runs out of the budget or successfully achieves the goal
        """

        # n_episodes = 200
        print("Number of episodes", n_episodes)

        min_exploration_rate = self.Q_function1.exploration_rate
        self.Q_function1.exploration_rate = 1.0
        self.Q_function2.exploration_rate = 1.0

        for episode in range(n_episodes):

            # self.Q_function1.exploration_rate = min_exploration_rate * (n_episodes / (episode + 1))
            # self.Q_function2.exploration_rate = min_exploration_rate * (n_episodes / (episode + 1))

            print("Episode", episode)
            print("Exploration rate", self.Q_function1.exploration_rate)

            all_rewards = []
            cumulative_reward = 0

            # Create a victim model and train
            victim_model = victim()
            ce_loss, reg = victim_model.train()
            fairness_loss = -1
            dp, eod = victim_model.evaluate()
            loss_val = reg * alpha
            # Result of majority
            # den0 = torch.sigmoid(output.view(-1))[self.sens == 0]
            # Result of minority
            # den1 = torch.sigmoid(output.view(-1))[self.sens == 1]

            emb_matrix = self.embedding.n2v(self.graph)
            state_embedding = self.embedding.g2v(emb_matrix)

            # self.metrics_logger.log_metrics(
            #     episode=episode + 1,
            #     iteration=0,
            #     accuracy=0,
            #     training_loss=0,
            #     demographic_parity=dp,
            #     equality_of_odds=eod,
            #     conditional_dp=cdp,
            #     surrogate_loss=fairness_loss
            # )
            # Launch a cycle of attack

            # Employ dynamic exploration rate to encourge more exploration during the previous stage

            for i in range(self.budegt):
                
                print(i,"th iteration")
                # Get the current state embedding
                # Select the first node:
                # How shall I make sure that the first node is different from the second node?
                
                first_node = self.Q_function1.select_action(state_embedding)
                second_node = self.Q_function2.select_action(state_embedding)

                # victim_model.change_edge(first_node, second_node)
                print("The selected two nodes are:", first_node, second_node)
                self.change_edge(first_node, second_node)
                victim_model.update_adj_matrix(torch.from_numpy(nx.to_numpy_array(self.graph)))

                # After changing the model, retrain the victim model and calculate the new fairness value
                ce_loss, reg = victim_model.train()
                dp, eod = victim_model.evaluate()
                new_loss_val =  reg * alpha
                # Determine the difference of fairness, which is the reward
                reward = new_loss_val - loss_val
                cumulative_reward += reward
                loss_val = new_loss_val

                # Update the embedding correspondingly as the new state
                emb_matrix = self.embedding.n2v(self.graph)
                new_state_embedding = self.embedding.g2v(emb_matrix).detach()

                self.Q_function1.replay_buffer.push(state=state_embedding.detach(), action=first_node, reward=reward,next_state=new_state_embedding, done=False)
                self.Q_function2.replay_buffer.push(state=state_embedding.detach(), action=second_node, reward=reward,next_state=new_state_embedding, done=False)
                state_embedding = new_state_embedding
                self.Q_function1.exploration_rate = min_exploration_rate
                self.Q_function2.exploration_rate = min_exploration_rate
                self.train_step()

            all_rewards.append(cumulative_reward)
            # Update the target network periodically
            if episode % 2 == 0:
                self.Q_function1.target_network.load_state_dict(self.Q_function1.policy_network.state_dict())
                self.Q_function2.target_network.load_state_dict(self.Q_function2.policy_network.state_dict())

            if (episode) % 1 == 0:
                avg_reward = np.mean(all_rewards[-10:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

        self.Q_function1.exploration_rate = min_exploration_rate
        self.Q_function2.exploration_rate = min_exploration_rate

    def train_step(self):
        """
        Train Q network periodically and minimize loss
        """
        self.Q_function1.train_step()
        self.Q_function2.train_step()

    def evaluate(self, epoch, alpha = 1):
        """
        Evaluate the trained model by setting exploration rate to be 0
        """

        # print("Episode", episode)
        print("Evaluation mode, the", epoch, "th epoch")
        self.Q_function1.exploration_rate = 0.0
        self.Q_function2.exploration_rate = 0.0
        min_exploration_rate = 0.0
        # print("Exploration rate", self.Q_function1.exploration_rate)

        all_rewards = []
        cumulative_reward = 0

        # Create a victim model and train
        victim_model = victim()
        ce_loss, reg = victim_model.train()
        dp,eod = victim_model.evaluate()
        emb_matrix = self.embedding.n2v(self.graph)
        state_embedding = self.embedding.g2v(emb_matrix)

        new_loss_val = reg * alpha
        # self.metrics_logger.log_metrics(
        #     episode= 1,
        #     iteration=0,
        #     accuracy=0,
        #     training_loss=0,
        #     demographic_parity=dp,
        #     equality_of_odds=eod,
        #     conditional_dp=cdp,
        #     surrogate_loss=fairness_loss
        # )
        # Launch a cycle of attack

        # Employ dynamic exploration rate to encourge more exploration during the previous stage

        for i in range(self.budegt):

            
            print(i,"th iteration")
            # Get the current state embedding
            # Select the first node:
            # How shall I make sure that the first node is different from the second node?
            
            first_node = self.Q_function1.select_action(state_embedding)
            second_node = self.Q_function2.select_action(state_embedding)

            # victim_model.change_edge(first_node, second_node)
            print("The selected two nodes are:", first_node, second_node)
            self.change_edge(first_node, second_node)
            victim_model.update_adj_matrix(torch.from_numpy(nx.to_numpy_array(self.graph)))

            # After changing the model, retrain the victim model and calculate the new fairness value
            victim_model.train()
            ce_loss, reg = victim_model.train()
            dp, eod = victim_model.evaluate()
            new_loss_val = reg * alpha
            # Determine the difference of fairness, which is the reward
            reward = new_loss_val - loss_val
            cumulative_reward += reward
            loss_val = new_loss_val

            # Update the embedding correspondingly as the new state
            emb_matrix = self.embedding.n2v(self.graph)
            new_state_embedding = self.embedding.g2v(emb_matrix).detach()

            # self.Q_function1.replay_buffer.push(state=state_embedding.detach(), action=first_node, reward=reward,next_state=new_state_embedding, done=False)
            # self.Q_function2.replay_buffer.push(state=state_embedding.detach(), action=second_node, reward=reward,next_state=new_state_embedding, done=False)
            state_embedding = new_state_embedding
            self.train_step()

            # self.metrics_logger.log_metrics(
            #     episode= 1,
            #     iteration=i + 1,
            #     accuracy=0.0,
            #     training_loss=0.0,
            #     demographic_parity=dp,
            #     equality_of_odds=eod,
            #     conditional_dp=cdp,
            #     surrogate_loss=fairness_loss
            # )

        avg_reward = np.mean(all_rewards[-10:])
        print(f"epoch {epoch}, Average Reward: {avg_reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

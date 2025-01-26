import torch.nn as nn
import torch
import networkx as nx
gpu_index = 2
class s2v_embedding(nn.Module):

    """
    structure to vector embedding class built from paper by Dai et al.
    Formula:
    u(v_i) = \sigma (W_1 x(u) + X_2 \Simga_{x \ in N(u)} \mu(u)^{k - 1})
    We use relu for non linear layer
    """

    def __init__(self, nnodes, feature_matrix, output_dim):
        """
        graph: networkx graph
        feature_matrix: torch tensor
        output_dim: int > 0
        """
        super(s2v_embedding, self).__init__()
        self.feature_matrix = feature_matrix # nnodes \times nfeatures
        self.output_dim = output_dim
        self.nfeatures = feature_matrix.shape[1]
        self.nnodes = nnodes

        # TODO
        # For here
        # print("nfeatures:", self.nfeatures)
        # print("output_dim:", self.output_dim)
        self.W1 = nn.Linear(in_features=self.nfeatures, out_features=self.output_dim, bias=True)
        self.W2 = nn.Linear(in_features=self.output_dim, out_features=self.output_dim, bias=True)
        # self.W1 = nn.Parameter(torch.randn(self.output_dim, self.nfeatures))
        # self.W2 = nn.Parameter(torch.randn(self.output_dim, self.output_dim))

        self.relu = nn.ReLU()
        # The result embedding matrix we are looking for
        

        self.reset_parameters()

        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        self.to(device)

    def reset_parameters(self):
        """
        Reset two weights using xavier uniform
        """
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def n2v(self, graph, T = 10):
        """
        Loop through nodes and create their new embeddings
        """
        device = torch.device(f"cuda:{gpu_index}"if torch.cuda.is_available() else "cpu")
        emb_matrix = torch.zeros(self.output_dim, self.nnodes)

        adjacency = nx.to_numpy_array(graph)  # [nnodes, nnodes]
        adjacency = torch.from_numpy(adjacency).float().to(device)  # Convert to torch tensor


        for _ in range(T):
            nbr_emb_sum = torch.matmul(adjacency, emb_matrix)  # [nnodes, output_dim]

            a = self.W1(self.feature_matrix)  # [nnodes, output_dim]
            b = self.W2(nbr_emb_sum)          # [nnodes, output_dim]
            new_embeddings = self.relu(a + b)  # [nnodes, output_dim]

            emb_matrix = new_embeddings

        return emb_matrix  # [nnodes, output_dim]

    def g2v(self, emb_matrix, node_list = None):
        """
        Following the paper, create graph embedding by summing up the node embedding
        """
        # emb_matrix = self.n2v(graph=graph)

        if node_list is None:
            node_list = [i for i in range(self.nnodes)]
        # print(node_list)
        # print(emb_matrix)
        graph_embedding = emb_matrix[node_list].sum(dim=0)  # [output_dim]

        return graph_embedding

    def get_graph_embeding(self, graph):

        emb_matrix = self.n2v(graph)          # [nnodes, output_dim]
        graph_embedding = self.g2v(emb_matrix)  # [output_dim]
        return graph_embedding
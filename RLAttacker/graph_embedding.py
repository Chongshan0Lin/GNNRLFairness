import torch.nn as nn
import torch
import networkx

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
        self.feature_matrix = feature_matrix
        self.output_dim = output_dim
        self.nfeatures = feature_matrix.shape[0]
        self.nnodes = nnodes

        self.W1 = nn.Linear(in_features=self.nfeatures, out_features=self.output_dim, bias=True)
        self.W2 = nn.Linear(in_features=self.output_dim, out_features=self.output_dim, bias=True)

        self.relu = True
        # The result embedding matrix we are looking for
        

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset two weights using xavier uniform
        """
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def n2v(self, graph, T = 10):
        """
        Loop through nodes and create their new embeddings
        """
        emb_matrix = torch.zeros(self.output_dim, self.nnodes)
        

        for _ in range(T):

            new_embeddings = torch.zeros(self.output_dim, self.nnodes)

            for node in range(self.nnodes):
                neighbors = graph.neighbors(node)
                nbr_emb_sum = sum(self.emb_matrix[neighbors])

                new_embeddings[node] = nn.ReLU(torch.matmul(self.W1, self.feature_matrix[node]) + torch.matmul(self.W2, nbr_emb_sum))

            emb_matrix = new_embeddings
        
        return emb_matrix

    def g2v(self, emb_matrix, node_list = None):
        """
        Following the paper, create graph embedding by summing up the node embedding
        """
        # emb_matrix = self.n2v(graph=graph)

        if node_list == None:
            node_list = [i for i in range(self.nnodes)]

        graph_embedding = sum(emb_matrix[node_list])

        return graph_embedding

    def get_graph_embeding(self, graph):

        embed_matrix = self.n2v(graph)
        return self.g2v(emb_matrix=embed_matrix)

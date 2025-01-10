import torch.nn as nn

"""
structure to vector embedding class built from paper by Dai et al.
"""
class s2v_embedding(nn.Module):
    def __init__(self, graph, feature_matrix, output_dim):
        super(s2v_embedding, self).__init__()
        self.graph = graph
        self.feature_matrix = feature_matrix
        self.output_dim = output_dim
        self.nfeatures = feature_matrix.shape[0]


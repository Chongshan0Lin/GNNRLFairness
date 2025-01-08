import os
import networkx as nx
import pandas as pd
import numpy as np

def graph_loading(path = "../Data/cora"):
    data_dir = os.path.expanduser(path)
    edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
    edgelist["label"] = "cites"
    edgelist.sample(frac=1).head(5)

    Graphnx = nx.from_pandas_edgelist(edgelist)
    nx.set_node_attributes(Graphnx, "Paper", "Label")

    return Graphnx

def feature_loading(path = "../Data/cora"):
    """
    Convert paper features into matrix in {0, 1}
    """    
    data_dir = os.path.expanduser(path)
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names =  feature_names + ["subject"]
    node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
    feature_matrix = node_data.to_numpy()[:, : -1]
    return feature_matrix

def label_loading(path = "../Data/cora"):
    """
    Convert paper features into matrix in {0, 1}
    """
    data_dir = os.path.expanduser(path)
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names =  feature_names + ["subject"]
    node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
    raw_label_matrix = node_data.to_numpy()[:, [-1]]
    label_mapping, label_matrix = np.unique(raw_label_matrix.T[0], return_inverse=True)
    
    return label_matrix, label_mapping







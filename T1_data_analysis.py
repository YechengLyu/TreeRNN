import os
from os import path
import numpy as np
from time import time
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


#%% global variables
dataset_name = os.path.basename(os.getcwd())
edges_path  = path.join(dataset_name,dataset_name+'_A.txt')
index_path  = path.join(dataset_name,dataset_name+'_graph_indicator.txt')
label_path  = path.join(dataset_name,dataset_name+'_graph_labels.txt')

#%% load the dataset
edges_raw = np.loadtxt(edges_path,delimiter=',').astype(np.int)
index_raw = np.loadtxt(index_path,delimiter=',').astype(np.int)
label_raw = np.loadtxt(label_path,delimiter=',').astype(np.int)

#%% parse the dataset
NUM_GRAPH = np.max(index_raw.max()-index_raw.min())+1
NUM_NODES = np.zeros(NUM_GRAPH,dtype=np.int)
NUM_EDGES = np.zeros(NUM_GRAPH,dtype=np.int)
NUM_DEGREE = np.zeros(NUM_GRAPH,dtype=np.int)
NUM_DEPTH = np.zeros(NUM_GRAPH,dtype=np.int)

for idx_graph in range(NUM_GRAPH):
    ##%% get the nodes count of the graph
    NUM_NODES[idx_graph] = np.count_nonzero(index_raw == idx_graph+index_raw.min())

    ##%% get the nodes of the graph
    nodes_graph_idx = np.where(index_raw == idx_graph+index_raw.min())[0]
    nodes_graph_min = nodes_graph_idx.min()+edges_raw.min()
    nodes_graph_max = nodes_graph_idx.max()+edges_raw.min()

    ##%% get the edges of the graph
    value = np.all([edges_raw[:,0] >= nodes_graph_min, edges_raw[:,0] <= nodes_graph_max],axis=0)
    NUM_EDGES[idx_graph] = np.count_nonzero(value)
    edges_graph = edges_raw[value]-nodes_graph_min

    ##%% build graph from nodes and edges
    data = np.ones(NUM_EDGES[idx_graph],dtype=np.int)
    G = csr_matrix((data,(edges_graph[:,0],edges_graph[:,1])))
    G_mat = G.toarray()
    G1 = nx.from_numpy_matrix(G_mat)

    ##%% check if graph is connected
    Flag_inf = nx.is_connected(G1)

    ##%% get the max degree and depth of the graph
    if(Flag_inf):
        NUM_DEGREE[idx_graph] = max(dict(G1.degree).values())
        NUM_DEPTH[idx_graph] = np.ceil(nx.diameter(G1)/2).astype(np.int)
    else:
        degree_list = []
        depth_list  = []
        G1_list = [c for c in sorted(nx.connected_components(G1), key=len, reverse=True)]
        for g in G1_list:
            G2 = G1.subgraph(g)
            degree_list.append(max(dict(G2.degree).values()))
            depth_list.append(np.ceil(nx.diameter(G2)/2).astype(np.int))
        NUM_DEGREE[idx_graph] = max(degree_list)
        NUM_DEPTH[idx_graph]  = max(depth_list)
        NUM_NODES[idx_graph] += len(G1_list)-1

    if(idx_graph%100 == 0):
        print(idx_graph,'/',NUM_GRAPH)
    # break

print('NUM_GRAPH',NUM_GRAPH,'NUM_DEGREE',NUM_DEGREE.max(),'NUM_DEPTH',NUM_DEPTH.max(),'NUM_NODES',NUM_NODES.max())
#%%
np.savetxt("statistics.csv", np.array([NUM_GRAPH,NUM_DEGREE.max(),NUM_DEPTH.max(),NUM_NODES.max()],dtype=np.int), delimiter=",",fmt='%d')

import os
from os import path
import numpy as np
from time import time
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import h5py
from tensorflow.keras.utils import to_categorical
#%% global variables
dataset_name = os.path.basename(os.getcwd())
myFile = np.genfromtxt('statistics.csv', delimiter=',', dtype=np.int)

NUM_GRAPH = myFile[0]
NUM_DEGREE= myFile[1]
NUM_DEPTH = myFile[2]
NUM_NODES = myFile[3]
NUM_REPEAT = 1

LEN_DATASET = NUM_GRAPH*NUM_REPEAT

hdf5_path = dataset_name+'test_'+str(NUM_REPEAT)+'x.hdf5'

#%% load the dataset
Flag_link = 0
Flag_node = 0

edges_path  = path.join(dataset_name,dataset_name+'_A.txt')
links_path  = path.join(dataset_name,dataset_name+'_edge_labels.txt')
nodes_path  = path.join(dataset_name,dataset_name+'_node_labels.txt')
index_path  = path.join(dataset_name,dataset_name+'_graph_indicator.txt')
label_path  = path.join(dataset_name,dataset_name+'_graph_labels.txt')

if(path.exists(links_path)):
    Flag_link = 1
if(path.exists(nodes_path)):
    Flag_node = 1


edges_raw = np.loadtxt(edges_path,delimiter=',').astype(np.int)
index_raw = np.loadtxt(index_path,delimiter=',').astype(np.int)
label_raw = np.loadtxt(label_path,delimiter=',').astype(np.int)
if(Flag_link):
    links_raw = np.loadtxt(links_path,delimiter=',').astype(np.int)
else:
    links_raw = np.ones((len(edges_raw)),dtype=np.int)

if(Flag_node):
    nodes_raw = np.loadtxt(nodes_path,delimiter=',').astype(np.int)
    nodes_raw -= nodes_raw.min()
else:
    nodes_raw = np.ones((len(index_raw)),dtype=np.int)

if(Flag_link):
    NUM_LINK_FE = (links_raw.max()-links_raw.min()+1)
else:
    NUM_LINK_FE = 2
if(Flag_node):
    NUM_NODE_FE = (nodes_raw.max()-nodes_raw.min()+1)
else:
    NUM_NODE_FE = 2

NUM_LAYER_FE = NUM_NODE_FE+NUM_LINK_FE
NUM_WIDTH   = NUM_NODES

##%% parse labels
labels = np.unique(label_raw)
NUM_LABELS  = len(labels)
label_raw_copy = np.copy(label_raw)
for i_label in range(NUM_LABELS):
    label_raw[label_raw_copy == labels[i_label]] = i_label


#%% define dataset
f = h5py.File(hdf5_path,'w')
x_set = f.create_dataset('x_train',shape=(LEN_DATASET,NUM_WIDTH,NUM_DEPTH+1,NUM_LAYER_FE),\
        dtype=np.float32,compression="lzf",chunks=(1,NUM_WIDTH,NUM_DEPTH+1,NUM_LAYER_FE))
y_set = f.create_dataset('y_train',shape=(LEN_DATASET,NUM_LABELS),\
        dtype=np.float32,compression="lzf",chunks=(1,NUM_LABELS))
t_set = f.create_dataset('t_train',shape=(LEN_DATASET,NUM_WIDTH,NUM_DEPTH+1),\
        dtype=np.float32,compression="lzf",chunks=(1,NUM_WIDTH,NUM_DEPTH+1))

#%% help function

def tree_size(G,source):
    return len(list(nx.bfs_edges(G,source)))+1
def bfs(G,root):
    # G = G1
    G_mat = nx.to_numpy_matrix(G)
    order1 = np.arange(len(G_mat))
    # np.random.shuffle(order1)
    
    
    G_mat1 = G_mat[order1] 
    G_mat1 = G_mat1[:,order1]
    
    order2 = np.argsort(order1)
    root1 = order2[root]
    ##%%
    G2 = nx.DiGraph()
    G2.add_nodes_from(np.arange(len(G_mat1)))
    arr_tra = np.zeros(len(G_mat),dtype = np.int)-1
    arr_tra[0] = root1
    idx_next = 1
    for idx_tra in range(len(arr_tra)):
        parent = arr_tra[idx_tra]
        children = np.where(G_mat1[parent])[1]
        children = np.setdiff1d(children,arr_tra)
        for child in children:
            G2.add_edge(parent,child)
        len_children = len(children)
        arr_tra[idx_next:idx_next+len_children] = children
        idx_next += len_children
        # break
    G2_mat =nx.to_numpy_matrix(G2)
    G2_mat1 = G2_mat[order2] 
    G2_mat1 = G2_mat1[:,order2]
    
    G3_edges = np.array(np.where(G2_mat1)).T
    G3 = nx.DiGraph()
    G3.add_nodes_from(np.arange(len(G_mat1)))
    G3.add_edges_from(G3_edges)
    return G3
#%% parse the dataset
idx_sample = 0
for idx_graph in range(NUM_GRAPH):

    for idx_repeat in range(NUM_REPEAT):
        ##%% get graph node idxs
        nodes_graph_idx = np.where(index_raw == idx_graph+index_raw.min())[0]
        nodes_graph_min = nodes_graph_idx.min()+edges_raw.min()
        nodes_graph_max = nodes_graph_idx.max()+edges_raw.min()

        ##%% get graph node features
        nodes_graph = nodes_raw[nodes_graph_idx]

        ##%% get graph edge idxs
        value = np.all([edges_raw[:,0] >= nodes_graph_min, edges_raw[:,0] <= nodes_graph_max],axis=0)
        edges_graph = edges_raw[value]-nodes_graph_min

        ##%% get graph edge features
        links_graph = links_raw[value]

        ##%% get graph labels
        label_graph = label_raw[idx_graph]

        ##%% build the graph in networkx
        G1 = nx.Graph()
        G1.add_nodes_from(np.arange(len(nodes_graph)))
        G1.add_edges_from(edges_graph)
        
        ##%% pre-store the graph layout for visulation
        # pos = nx.kamada_kawai_layout(G1)
        # pos = np.array(list(pos.values()))

        Flag_connected = nx.is_connected(G1)
        
        ##%% adjust the layout to avoid overlaps
        # if not Flag_connected:
        #     G1_list = [c for c in sorted(nx.connected_components(G1), key=len, reverse=True)]
        #     boundary = np.array([0,0])
        #     for g in G1_list:
        #         G2 = G1.subgraph(g)   
        #         G2_nodes = list(G2.nodes)
        #         pos[G2_nodes] = pos[G2_nodes] - pos[G2_nodes].min(0) + boundary + 0.3
        #         boundary = pos[G2_nodes].max(0)
        
        ##%% draw the original graph
        # plt.figure()
        # nx.draw(G1,pos=pos)
        # nx.draw_networkx_labels(G1,pos=pos)


        ##%% build Breadth First Search (bfs) Tree from the graph
        if Flag_connected:
            root_candidate = nx.center(G1)
            # np.random.shuffle(root_candidate)
            root = root_candidate[0:1]
            G3 = bfs(G1,root[0])
        
        else:
            G1_list = [c for c in sorted(nx.connected_components(G1), key=len, reverse=True)]
            root = []
            G3 = nx.DiGraph()
            G3.add_nodes_from(np.arange(G1.number_of_nodes()))
            for g in G1_list:
                G2 = G1.subgraph(g)  
                order1 = list(G2.nodes)
                G2_mat = nx.to_numpy_matrix(G2)
                G2 = nx.from_numpy_matrix(G2_mat)
                root_candidate = nx.center(G2)
                # np.random.shuffle(root_candidate)
                root1 = root_candidate[0]
                root.append(order1[root1])
                G2 = bfs(G2,root1)
                G2_edges = np.array(list(G2.edges()))
                G3_edges = G2_edges.copy()
                for idx in range(G2.number_of_nodes()):
                    G3_edges[G2_edges==idx]=order1[idx]
                G3.add_edges_from(G3_edges)
            
        ##%% draw the bfs tree
        # plt.figure()
        # nx.draw(G3,pos=pos)
        # nx.draw_networkx_labels(G3,pos=pos)

        ##%% pre-define the block space for the bfs tree
        tree = np.zeros((NUM_NODES,NUM_DEPTH+1),dtype=np.int)-1
        idx_col = 0
        for leaf in root:
            leaf_size = tree_size(G3,leaf)
            tree[idx_col:idx_col+leaf_size,-1] = leaf
            idx_col += leaf_size
        
        ##%% transform the bfs tree to block format
        for i_depth in range(1,NUM_DEPTH+1):
            ##%% for each depth, traverse every node and get the leaves
            ##%% for each root node, reserve a row in the block
        
            root1 = []
            for idx_root in range(len(root)):
                node = root[idx_root]
                if(node>=0):
                    leaf = list(G3[node])
                    # np.random.shuffle(leaf)
                else:
                    leaf = []
                root1 += leaf
                root1 += [-1]
        
            ##%% write the leaves of current depth to the block
            idx_row = 0
            for leaf in root1:
        
                if(leaf==-1):
                    ##%% for each reserved node, skip the space
                    idx_row +=1
                    continue
                else:
                    ##%% for each valid node, write the block with
                    ##%% corresponding width that equals to the successor size
                    leaf_size = tree_size(G3,leaf)
                    tree[idx_row:idx_row+leaf_size,-1-i_depth] = leaf
                    idx_row += leaf_size
        
            root = root1

        ##%% initialize the feature matrix
        mat = np.zeros_like(x_set[0])

        ##%% write the node features and edge features to the matrix
        for i_row in range(NUM_NODES):
            for i_col in range(NUM_DEPTH+1):
                node = tree[i_row,i_col]
        
                ##%% skip reserved node spaces
                if(node==-1):
                    continue
                else:
                    ##%% get node features
                    node_feature = \
                                to_categorical(nodes_graph[node],NUM_NODE_FE)
        
                    ##%% get edge by searching the edge list
                    if(i_col == NUM_DEPTH):
                        edge_feature = np.zeros((NUM_LINK_FE))
                    else:
                        predecessor = tree[i_row,i_col+1]
        
                        if(node != predecessor):
                            value = np.all([edges_graph[:,0] == predecessor, \
                                            edges_graph[:,1] == node],axis=0)
                            edge_idx = np.where(value)[0][0]
        
                            ##%% get edge features
                            edge_feature = \
                                to_categorical(links_graph[edge_idx],NUM_LINK_FE)
                        else:
                            edge_feature = np.zeros((NUM_LINK_FE))
        
                    block_feature = np.hstack((node_feature,edge_feature))
                    mat[i_row,i_col,:] = block_feature


        ##%% wrtie feature matrix to dataset
        x_set[idx_sample] = mat

        label_sample = to_categorical(label_graph>0,NUM_LABELS)
        y_set[idx_sample] = label_sample

        t_set[idx_sample] = tree

        idx_sample += 1
        print(idx_sample,'/',LEN_DATASET)


#%%
f.close()

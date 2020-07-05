import os
from os import path
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
#%%
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
#%% global settings
idx_fold = 0

#%% global variables
dataset_name = os.path.basename(os.getcwd())
myFile = np.genfromtxt('statistics.csv', delimiter=',', dtype=np.int)

NUM_GRAPH = myFile[0]
NUM_DEGREE= myFile[1]
NUM_DEPTH = myFile[2]
NUM_NODES = myFile[3]

NUM_EPOCHS = 1000

# hdf5_train_path = dataset_name+'train_'+str(NUM_REPEAT)+'x.hdf5'
# hdf5_test_path = hdf5_train_path
# hdf5_test_path  = dataset_name+'test_1x.hdf5'
#%% load dataset
# f_train = h5py.File(hdf5_train_path,'r')
# x_train = f_train['x_train'][:]
# y_train = f_train['y_train'][:]
# t_train = f_train['t_train'][:]

# f_test = h5py.File(hdf5_test_path,'r')
# x_test = f_test['x_train']
# y_test = f_test['y_train']
# t_test = f_test['t_train']
#%% split training and testing set
NUM_GRAPH_TRAIN = NUM_GRAPH
NUM_FOLD_TRAIN = NUM_GRAPH_TRAIN/10.
NUM_GRAPH_TEST = NUM_GRAPH
NUM_FOLD_TEST = NUM_GRAPH_TEST/10.

idx_train = np.arange(np.int(NUM_FOLD_TRAIN*idx_fold),np.int(NUM_FOLD_TRAIN*(idx_fold+1)))
idx_train = np.delete(np.arange(NUM_GRAPH_TRAIN),idx_train)
idx_test = np.arange(np.int(NUM_FOLD_TEST*idx_fold),np.int(NUM_FOLD_TEST*(idx_fold+1)))

#%%
# class Dataloder(keras.utils.Sequence):
#     def __init__(self, x_train_set,y_train_set,idxs_list, batch_size=1, shuffle=False):
#         self.x_train_set = x_train_set
#         self.y_train_set = y_train_set
#         # self.t_train_set = t_train_set
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indexes = idxs_list
#         self.on_epoch_end()
#         self.x_shape = list(x_train_set.shape)
#         self.y_shape = list(y_train_set.shape)

#     def __getitem__(self, i):
#         # collect batch data
#         start = i * self.batch_size
#         stop = (i + 1) * self.batch_size
#         data_x_shape = self.x_shape
#         data_x_shape[0] = self.batch_size
#         data_x = np.zeros(data_x_shape,dtype=np.float32)

#         data_y_shape = self.y_shape
#         data_y_shape[0] = self.batch_size
#         data_y = np.zeros(data_y_shape,dtype=np.float32)

#         for j in range(stop-start):
#             data_x[j] = self.x_train_set[self.indexes[start+j]]
#             data_y[j] = self.y_train_set[self.indexes[start+j]]
#             # data_t[j] = self.t_train_set[self.indexes[start+j]]

#         # data_x = np.flip(data_x,1)
#         return data_x,data_y

#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return len(self.indexes) // self.batch_size

#     def on_epoch_end(self):
#         """Callback function to shuffle indexes each epoch"""
#         if self.shuffle:
#             self.indexes = np.random.permutation(self.indexes)

#%% initialize the data loader
# train_loader = Dataloder(x_train,y_train,t_train,idx_train,batch_size=64,shuffle=True)
# test_loader  = Dataloder(x_test,y_test,idx_test,batch_size=1)

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


#%% help function
def tree_size(G,source):
    return len(list(nx.bfs_edges(G,source)))+1
def bfs(G,root):
    # G = G1
    G_mat = nx.to_numpy_matrix(G)
    order1 = np.arange(len(G_mat))
    np.random.shuffle(order1)
    
    
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

#%%
class trainsetloder(keras.utils.Sequence):
    def __init__(self, idxs_list, batch_size=1, shuffle=False):
        self.x_set = np.zeros((batch_size,NUM_WIDTH,NUM_DEPTH+1,NUM_LAYER_FE),dtype = np.float32)
        self.y_set = np.zeros((batch_size,NUM_LABELS),dtype=np.float32)
        self.t_set = np.zeros((batch_size,NUM_WIDTH,NUM_DEPTH+1),dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = idxs_list
        self.on_epoch_end()
        self.x_shape = list(self.x_set.shape)
        self.y_shape = list(self.y_set.shape)
        
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
        
    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        # data_x_shape = self.x_shape
        # data_x_shape[0] = self.batch_size
        # data_x = np.zeros(data_x_shape,dtype=np.float32)

        # data_y_shape = self.y_shape
        # data_y_shape[0] = self.batch_size
        # data_y = np.zeros(data_y_shape,dtype=np.float32)

        for j in range(stop-start):
            idx_graph = self.indexes[start+j]
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
                np.random.shuffle(root_candidate)
                root = root_candidate[0:1]
                G3 = bfs(G1,root[0])
                # G3 = nx.bfs_tree(G1,root[0])
            
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
                    np.random.shuffle(root_candidate)
                    root1 = root_candidate[0]
                    root.append(order1[root1])
                    G2 = bfs(G2,root1)
                    # G2 = nx.bfs_tree(G2,root1)
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
                        np.random.shuffle(leaf)
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
            mat = np.zeros_like(self.x_set[0])-1
    
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
            self.x_set[j] = mat
    
            label_sample = to_categorical(label_graph>0,NUM_LABELS)
            self.y_set[j] = label_sample

        # data_x = np.flip(data_x,1)
        return self.x_set,self.y_set

        
#%%
train_loader = trainsetloder(idx_train, batch_size=64, shuffle=True)
test_loader = trainsetloder(idx_test, batch_size=1, shuffle=False)

        
#%% define TreeRNN layer

def TreeRNN(inputs,filters):
    input_shape = inputs.shape[1:].as_list()
    input_tensor = L.Input(shape=input_shape)
    # LSTM1 = L.Bidirectional(L.SimpleRNN(filters,activation = 'tanh',return_sequences=True,unroll=True,name='LSTM'))
    # LSTM1 = L.LSTM(filters,return_sequences=True)
    # LSTM1 = L.Bidirectional(L.LSTM(filters,return_sequences=True))
    LSTM1 = L.SimpleRNN(filters,activation = 'tanh',return_sequences=True,unroll=True)
    # LSTM1 = L.Conv1D(filters,3,padding='same',activation='tanh')
    x = input_tensor[:,:,0,:]

    for i_depth in range(1,input_shape[1]):
        x1 = input_tensor[:,:,i_depth,:]
        x = L.Concatenate(axis=-1)([x,x1])
        x = LSTM1(x)
        # x2 = 1-tf.reduce_max(x,keepdims=True,axis=-1)
        # x2 = tf.tile(x2,[1,1,filters])
        # x = x*x2

    model = keras.Model(inputs=input_tensor, outputs=x,name='TreeRNN')
    return model

# %% define network
MLP2  = L.Conv2D(64,(1,1),padding='valid',activation='tanh',name='MLP1')
inputs1 = L.Input(shape = train_loader.x_set.shape[1:],name='Input')



x = inputs1
# x = L.Masking(mask_value=-1.,name='mask') (x)
x = MLP2(x)
 
x = TreeRNN(x,64)(x)
# x = L.Dense(16,activation='tanh',name='MLP2')(x)
# x = L.TimeDistributed(L.SimpleRNN(32,activation = 'tanh',return_sequences=True,unroll=True),name='LSTM')(x)

# x2 = 1-tf.reduce_max(x,keepdims=True,axis=-1)
# x2 = tf.tile(x2,[1,1,1,32])
# x = x*x2

x = L.GlobalMaxPooling1D()(x)

x = L.Dense(2,activation='softmax',name='mapping')(x)

outputs = x
model = keras.Model(inputs = inputs1, outputs = outputs)
model.summary()

opti = keras.optimizers.Adam(1e-3)
model.compile(opti,loss ='categorical_crossentropy',metrics=['acc'])
# model.load_weights('model_fold_'+str(idx_fold)+'.h5')
#%%
ckpt = keras.callbacks.ModelCheckpoint(filepath = 'model_fold_'+str(idx_fold)+'.h5',save_weights_only=False,monitor='val_acc')
history = model.fit(x=train_loader, validation_data=test_loader,
            epochs = NUM_EPOCHS,  verbose=1,callbacks=[ckpt])
##%%
##%%
acc = np.array(history.history['val_acc'])
acc_unique,count = np.unique(acc,return_counts=True)
for a,c in zip(acc_unique,count):
    print('acc:',a,', count:',c)
#%%
# f_train.close()
# f_test.close()

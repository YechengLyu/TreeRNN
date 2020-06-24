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
#%% global settings
idx_fold = 0

#%% global variables
dataset_name = os.path.basename(os.getcwd())
myFile = np.genfromtxt('statistics.csv', delimiter=',', dtype=np.int)

NUM_GRAPH = myFile[0]
NUM_DEGREE= myFile[1]
NUM_DEPTH = myFile[2]
NUM_NODES = myFile[3]

NUM_REPEAT = 1000

hdf5_train_path = dataset_name+'train_'+str(NUM_REPEAT)+'x.hdf5'
# hdf5_test_path = hdf5_train_path
hdf5_test_path  = dataset_name+'test_1x.hdf5'
#%% load dataset
f_train = h5py.File(hdf5_train_path,'r')
x_train = f_train['x_train'][:]
y_train = f_train['y_train'][:]
t_train = f_train['t_train'][:]

f_test = h5py.File(hdf5_test_path,'r')
x_test = f_test['x_train'][:]
y_test = f_test['y_train'][:]
t_test = f_test['t_train'][:]
#%% split training and testing set
NUM_GRAPH_TRAIN = len(x_train)
NUM_FOLD_TRAIN = NUM_GRAPH_TRAIN/10.
NUM_GRAPH_TEST = len(x_test)
NUM_FOLD_TEST = NUM_GRAPH_TEST/10.

idx_train = np.arange(np.int(NUM_FOLD_TRAIN*idx_fold),np.int(NUM_FOLD_TRAIN*(idx_fold+1)))
idx_train = np.delete(np.arange(NUM_GRAPH_TRAIN),idx_train)
idx_test = np.arange(np.int(NUM_FOLD_TEST*idx_fold),np.int(NUM_FOLD_TEST*(idx_fold+1)))

#%%
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, x_train_set,y_train_set,t_train_set,idxs_list, batch_size=1, shuffle=False):
        self.x_train_set = x_train_set
        self.y_train_set = y_train_set
        self.t_train_set = t_train_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = idxs_list
        self.on_epoch_end()
        self.x_shape = list(x_train_set.shape)
        self.y_shape = list(y_train_set.shape)

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data_x_shape = self.x_shape
        data_x_shape[0] = self.batch_size
        data_x = np.zeros(data_x_shape,dtype=np.float32)

        data_y_shape = self.y_shape
        data_y_shape[0] = self.batch_size
        data_y = np.zeros(data_y_shape,dtype=np.float32)

        for j in range(stop-start):
            data_x[j] = self.x_train_set[self.indexes[start+j]]
            data_y[j] = self.y_train_set[self.indexes[start+j]]
            # data_t[j] = self.t_train_set[self.indexes[start+j]]

        data_x = np.flip(data_x,1)
        return data_x,data_y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

#%% initialize the data loader
train_loader = Dataloder(x_train,y_train,t_train,idx_train,batch_size=64,shuffle=True)
test_loader  = Dataloder(x_test,y_test,t_test,idx_test,batch_size=1)


#%% define TreeRNN layer

def TreeRNN(inputs,filters):
    input_shape = inputs.shape[1:].as_list()
    input_tensor = L.Input(shape=input_shape)
    LSTM1 = L.SimpleRNN(filters,activation = 'tanh',return_sequences=True,unroll=True,name='LSTM')
    # LSTM1 = L.LSTM(filters,return_sequences=True)
    x = input_tensor[:,:,0,:]

    for i_depth in range(1,input_shape[1]):
        x1 = input_tensor[:,:,i_depth,:]
        x = L.Concatenate(axis=-1)([x,x1])
        x = LSTM1(x)

    model = keras.Model(inputs=input_tensor, outputs=x,name='TreeRNN')
    return model
# %% define network
MLP2  = L.Conv2D(32,(1,2),padding='valid',activation='tanh',name='MLP2')

inputs1 = L.Input(shape = list(x_train.shape)[1:],name='Input')

x = inputs1
x = MLP2(x)

x = TreeRNN(x,32)(x)

x = L.GlobalMaxPooling1D()(x)

x = L.Dense(2,activation='softmax',name='mapping')(x)

outputs = x
model = keras.Model(inputs = inputs1, outputs = outputs)
model.summary()

opti = keras.optimizers.Adam(1e-3)
model.compile(opti,loss ='categorical_crossentropy',metrics=['acc'])
#%%
ckpt = keras.callbacks.ModelCheckpoint(filepath = 'model.h5',save_weights_only=False,monitor='val_acc')
history = model.fit(x=train_loader, validation_data=test_loader,
          shuffle="batch", epochs = 100,  verbose=1,callbacks = [ckpt])
##%%
acc = np.array(history.history['val_acc'])
print('iter',acc.argmax(),', acc =',acc.max())
#%%
f_train.close()
f_test.close()
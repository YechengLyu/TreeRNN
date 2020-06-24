# TreeRNN

Tree is a powerful topology-preserving deep graph embedding and larning framework for graph classification. If you are interested in our work, please reference our paper on Arxiv (https://arxiv.org/abs/2006.11825).

## Installation
```
conda create -n TreeRNN python=3.7
conda activate TreeRNN
conda install tensorflow-gpu numpy scipy networkx h5py
git clone https://github.com/YechengLyu/TreeRNN.git
cd TreeRNN
```
 
 ## Training & Testing
#1 To train and test using TreeRNN, we firstly need to choose the dataset we aim to. Open the file "T0_download_dataset.py" and turn to Line 5, change the 'MUTAG' to any dataset avaliable at https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets. Then run the following command:
```
python T0_download_dataset.py
```
The code will download the dataset, analysis it and record the statistics including the number of graphs as well as maximum graph size, ridius and node degree.

#2 To prepare the dataset, open the file "T2_prepare_dataset_training.py" in the corresponding dataset folder and change the data augmentation number at Line 18. Currently it is set to 1000. Numbers less than 1000 are reasonable. After setting the data augmentation rate, run the following commands:

```
python T2_prepare_dataset_testing.py
python T2_prepare_dataset_training.py
```

 #3 To train the network, open the file "T3_training.py" and change the NUM_REPEAT at line 28 to the augmentation rate in step 2. Then change the idx_fold from 0 to 9 to select the fold to train and test. After the settings, run the following commend:
 
```
python T3_training.py
```

After training each fold, the best best validation accuracy will show in the commend line.

## Contact
If you have any questions, please feel free to contact Yecheng Lyu at ylyu@wpi.edu

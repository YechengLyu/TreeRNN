import os
import time
#%% global setting
dataset_link_prefix = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/'
dataset_name = 'MUTAG'
#%% download dataset
os.system('mkdir '+dataset_name)
os.system('wget '+dataset_link_prefix+dataset_name+'.zip')
os.system('mv '+dataset_name+'.zip '+dataset_name+'/')
os.system('unzip '+dataset_name+'/'+dataset_name+'.zip -d '+dataset_name+'/')
#%% copy codes to dataset folder
os.system('cp T1_data_analysis.py '+dataset_name+'/')
os.system('cp T3_training_EndToEnd.py '+dataset_name+'/')

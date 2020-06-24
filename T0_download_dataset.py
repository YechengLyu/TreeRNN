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
#%% analysis dataset
os.system('cp T1_data_analysis.py '+dataset_name+'/')
os.chdir(dataset_name)
os.system('python '+'T1_data_analysis.py')
os.chdir('..')

#%% copy codes to dataset folder
os.system('cp T2_prepare_dataset_training.py '+dataset_name+'/')
os.system('cp T2_prepare_dataset_testing.py '+dataset_name+'/')
os.system('cp T3_training.py '+dataset_name+'/')

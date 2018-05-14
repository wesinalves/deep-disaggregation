'''Trains an GRU model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
'''
from __future__ import print_function

from keras.models import Sequential
from keras.layers import LSTM,Conv1D, Bidirectional, Dense, Flatten
import numpy as np
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot  as plt
import time
import gc
import os
import psutil

p = psutil.Process(os.getpid())

def disag_error(y_true,y_pred):
  return K.sum(K.square(K.abs(y_pred - y_true)), axis = -1) / 2



appliances_name = ['north_bedroom', 'mastersouth_bedroom', 'basement_plugsligths', 'clothes_dryer','clothes_washer',
        'dining_roomplugs', 'dishwasher','eletronic_workbench', 'security_equipament', 'kitchen_fridge',
        'hvac', 'garage', 'heat_pump', 'hot_water', 'home_office','outside_plug','entertainment',
        'utility_plug','wall_oven']

batch_size = 150
epochs = 20
window_size = 60



f = '../../dataset/window_stride/cost/input_signal60.csv'
input_signal = np.loadtxt(f, delimiter='\t')
input_test = input_signal[((input_signal.shape[0]//3)*2):input_signal.shape[0],:]
input_test = input_test.reshape(input_test.shape[0],1,input_test.shape[1])

cpu_time = np.zeros(len(appliances_name))
memory_cost = np.zeros(len(appliances_name))

#load_time = []
#previous_time = 0
#load_time.append(previous_time + (time.time() - l_time))
#previous_time = previous_time + l_time

for i in range(len(appliances_name)):
  start_time = time.time()
  print('Calculating runtime round {0}...'.format(i))
  total_memory = 0
  for n in range(i+1):
    f = '../../dataset/window_stride/cost/output_signal_'+appliances_name[n]+'.csv'
    output_signal = np.loadtxt(f, delimiter='\t')
    #output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0],:]
    output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0]]
    output_test = output_test.reshape(output_test.shape[0],1,1)
    #output_test = output_test.reshape(output_test.shape[0],1,1)
    model = load_model('conv_zhang2_'+appliances_name[n]+'.h5', custom_objects={'disag_error':disag_error})
    mse, mae, disag = model.evaluate(input_test, output_test, batch_size=batch_size)
    output_signal = None
    output_test = None
    model = None
    gc.collect()
    K.reset_uids()
    K.clear_session()
    total_memory += p.memory_info().rss
    print("rss={}MB\tvms={}MB".format(p.memory_info().rss//10**6,p.memory_info().vms//10**6))
    
  cpu_time[i] =  (time.time() - start_time)
  memory_cost[i] = total_memory

print(cpu_time)
print(memory_cost)
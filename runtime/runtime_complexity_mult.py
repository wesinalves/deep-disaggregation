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


start_time  = time.time()
load_memory = 0
f = '../../dataset/multivariate/input_signal08std.csv'
input_signal = np.loadtxt(f, delimiter='\t')
f = '../../dataset/multivariate/output_signal08std.csv'
output_signal = np.loadtxt(f, delimiter='\t')


load_time = time.time() - start_time
load_memory = p.memory_info().rss

batch_size = 255
timesteps = 60
input_test = input_signal[((input_signal.shape[0]//3)*2):input_signal.shape[0],:]
num_features = input_test.shape[1]
input_test = input_test.reshape(input_test.shape[0]//timesteps,timesteps,input_test.shape[1])

cpu_time = np.zeros(len(appliances_name))
memory_cost = np.zeros(len(appliances_name))  




for n in range(19):
  total_memory = 0
  start_time = time.time()
  print('Calculating runtime round {0}...'.format(n))
  output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0],0:n+1]
  num_outputs = n+1
  output_test = output_test.reshape(output_test.shape[0]//timesteps,timesteps,num_outputs)
  model = load_model('gru_w{}.h5'.format(n+1), custom_objects={'disag_error':disag_error})
  mse, mae, disag = model.evaluate(input_test, output_test, batch_size=batch_size)
  output_test = None
  model = None
  gc.collect()
  K.reset_uids()
  K.clear_session()
  total_memory = p.memory_info().rss
  print("rss={}MB\tvms={}MB".format(p.memory_info().rss//10**6,p.memory_info().vms//10**6))  
  cpu_time[n] = (load_time/(19-n)) + (time.time() - start_time)
  memory_cost[n] = (load_memory/(19-n)) + total_memory

print(cpu_time)
print(memory_cost)
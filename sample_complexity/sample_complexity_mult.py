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


def disag_error(y_true,y_pred):
  return K.sum(K.square(K.abs(y_pred - y_true)), axis = -1) / 2

def accuracy(y_pred, y_true):
  return 1 - (np.sum(np.fabs(y_pred-y_true))/(2*np.sum(np.fabs(y_true))))

start_time  = time.time()
load_memory = 0
f = '../../dataset/multivariate/input_signal07std.csv'
input_signal = np.loadtxt(f, delimiter='\t')
f = '../../dataset/multivariate/output_signal07std.csv'
output_signal = np.loadtxt(f, delimiter='\t')


load_time = time.time() - start_time
load_memory = p.memory_info().rss

batch_size = 255
timesteps = 60
input_test = input_signal[((input_signal.shape[0]//100)*80):input_signal.shape[0],:]
num_features = input_test.shape[1]
input_test = input_test.reshape(input_test.shape[0]//timesteps,timesteps,num_features)

output_test = output_signal[((output_signal.shape[0]//100)*80):output_signal.shape[0],:]
num_outputs = output_test.shape[1]
output_test = output_test.reshape(output_test.shape[0]//timesteps,timesteps,num_outputs)

num_testsamples = output_test.shape[0]*timesteps

errors = []
accuracies = []


for n in range(80):
  print('Calculating sample complexity round {0}...'.format(n))  
  model = load_model('grumodel/gru_w{}.h5'.format(n), custom_objects={'disag_error':disag_error})
  mse, mae, disag = model.evaluate(input_test, output_test, batch_size=batch_size)
  target = model.predict(input_test)
  acc = accuracy(target.reshape(num_testsamples,num_outputs),output_test.reshape(num_testsamples,num_outputs))
  model = None
  gc.collect()
  K.reset_uids()
  K.clear_session()
  errors.append(mse)
  accuracies.append(acc)

print(errors)
print(accuracies)
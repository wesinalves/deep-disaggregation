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
start_time = time.time()

def disag_error(y_true,y_pred):
  return K.sum(K.square(K.abs(y_pred - y_true)), axis = -1) / 2

def accuracy(y_pred, y_true):
  return 1 - (np.sum(np.fabs(y_pred-y_true))/(2*np.sum(np.fabs(y_true))))

print('Loading data...')
with open('../dataset/multivariate/input_signal07std.csv', 'rb') as f:
  input_signal = np.loadtxt(f, delimiter='\t')
  f.close()
with open('../dataset/multivariate/output_signal07std.csv', 'rb') as f:
  output_signal = np.loadtxt(f, delimiter='\t')
  f.close()

appliances_name = ['clothes','dish','fridge','hvac', 'hotter']

input_test = input_signal[((input_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),:]
output_test = output_signal[((output_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),:]

batch_size = 255
timesteps = 60
num_features = input_test.shape[1]
num_outputs = output_test.shape[1]


input_test = input_test.reshape(input_test.shape[0]//timesteps,timesteps,num_features)
output_test = output_test.reshape(output_test.shape[0]//timesteps,timesteps,num_outputs)


#num_testsamples = output_test.shape[0]*timesteps



model = load_model('gru15.h5', custom_objects={'disag_error':disag_error})

target = model.predict(input_test)
#acc = accuracy(target.reshape(num_testsamples,num_outputs),output_test.reshape(num_testsamples,num_outputs))

#print(acc)

#print(target[0][:,2])
#print(output_test[0][:,2])
'''
y_pred = target[17][:,2]
y = output_test[17][:,2]
plt.plot(y_pred, color='b', label='estimated')
plt.plot(y, color='k', label='ground_truth')
plt.xlabel("Samples")
plt.ylabel("Fridge")

plt.legend()
plt.show()
'''
opacity = 0.4

for i in range(5):
  ax = plt.subplot(5,1,i+1)
  y_pred = target[17][:,i]
  y = output_test[17][:,i]
  ax.set_ylabel(appliances_name[i])
  plt.plot(y_pred, 'yo-', label='estimated', alpha=opacity)
  plt.plot(y, 'k--', label='ground_truth', alpha=opacity)

plt.legend()
plt.show()
'''
path = '../dataset/multivariate/output_dolly.csv'
np.savetxt(path, output_test, delimiter='\t', fmt='%s')
path = '../dataset/multivariate/input_dolly.csv'
np.savetxt(path, input_test, delimiter='\t', fmt='%s')
'''
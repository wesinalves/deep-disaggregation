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

def disag_error(y_true,y_pred):
  return K.sum(K.square(K.abs(y_pred - y_true))) / 2

def accuracy(y_pred, y_true):
  return 1 - (np.sum(np.fabs(y_pred-y_true))/(2*np.sum(np.fabs(y_true))))

batch_size = 150
epochs = 20
window_size = 60

'''
print('Loading data...')
with open('../dataset/window_stride/input_signal60.csv', 'rb') as f:
  input_signal = np.loadtxt(f, delimiter='\t')
  f.close()
with open('../dataset/window_stride/output_signal60_kitchen_fridge.csv', 'rb') as f:
  output_signal = np.loadtxt(f, delimiter='\t')
  f.close()
'''
print('Loading data...')
with open('../dataset/multivariate/input_signal07std.csv', 'rb') as f:
  input_signal = np.loadtxt(f, delimiter='\t')
  f.close()
with open('../dataset/multivariate/output_signal07std.csv', 'rb') as f:
  output_signal = np.loadtxt(f, delimiter='\t')
  f.close()


input_test = input_signal[((input_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),0]
output_test = output_signal[((input_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),2]
#output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0]]

#num_outputs = output_test.shape[1]
#num_outputs = 1
#num_testsamples = output_test.shape[0]
timesteps = 60

input_test = input_test.reshape(input_test.shape[0]//timesteps,1,timesteps)
output_test = output_test.reshape(output_test.shape[0]//timesteps,1,timesteps)
#output_test = output_test.reshape(output_test.shape[0],1,1)

path = '../dataset/window_stride/output_dolly.csv'
np.savetxt(path, output_test, delimiter='\t', fmt='%s')
path = '../dataset/window_stride/input_dolly.csv'
np.savetxt(path, input_test, delimiter='\t', fmt='%s')
# load Model
#print('load model...')

#model = load_model('conv_zhang1e.h5', custom_objects={'disag_error':disag_error})


#target = model.predict(input_test)

#acc = accuracy(target.reshape(num_testsamples,num_outputs),output_test.reshape(num_testsamples,num_outputs))

#print(acc)

'''
model = load_model('lstm_kelly5_withstride.h5', custom_objects={'disag_error':disag_error})

model.summary()

mse, mae, disag = model.evaluate(input_test, output_test,
                            batch_size=batch_size)
print('Test mse:', mse)
print('Test mae:', mae)
print('Test disag:', disag)

target = model.predict(input_test[10].reshape(1,1,60))

print(target[0][0])
print(output_test[10][0])
#y_pred = target[0][0]
for i in range(10):
  ax = plt.subplot(10,1,i+1)
  y = output_test[i + 80][0]
  #plt.plot(y_pred, color='b', label='estimated')
  plt.plot(y, color='k', label='real')
'''
for i in range(10):
	ax = plt.subplot(10,1,i+1)
	#y_pred = model.predict(input_test[i*12 + 60][0].reshape(1,1,60))[0][0]
	y = output_test[i][0]
	#plt.plot(y_pred, color='b', label='estimated')
	plt.plot(y, color='k', label='real')

plt.legend()
plt.show()

y_pred = target[17][0][0]
y = output_test[17][0]
plt.plot(y_pred, color='b', label='estimated')
plt.plot(y, color='k', label='ground_truth')
plt.xlabel("Samples")
plt.ylabel("Fridge")


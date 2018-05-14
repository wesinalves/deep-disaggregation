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

models = ['conv_zhang1b.h5', 'conv_zhang1a.h5', 
    'conv_zhang1c.h5','conv_zhang1d.h5','conv_zhang1e.h5']

datasets = ['dishwasher','clothes_washer','hot_water','hvac','kitchen_fridge']

batch_size = 150
epochs = 20
window_size = 60

accs = np.zeros(5)
disags = np.zeros(5)

print('Loading data...')
with open('../dataset/window_stride/input_signal60.csv', 'rb') as f:
  input_signal = np.loadtxt(f, delimiter='\t')
  f.close()



for i in range(5):
  print('Loading data...')
  with open('../dataset/window_stride/output_signal60_'+datasets[i]+'.csv', 'rb') as f:
    output_signal = np.loadtxt(f, delimiter='\t')
    f.close()
  # split data in train and test data
  input_train = input_signal[0:((input_signal.shape[0]//3)*2),:]
  output_train = output_signal[0:((output_signal.shape[0]//3)*2),:]
  #output_train = output_signal[0:((output_signal.shape[0]//3)*2)]

  input_test = input_signal[((input_signal.shape[0]//3)*2):input_signal.shape[0],:]
  output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0],:]
  #output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0]]

  num_outputs = output_train.shape[1]
  #num_outputs = 1
  num_testsamples = output_test.shape[0]

  # reshape data for lstm network
  input_train = input_train.reshape(input_train.shape[0],1,input_train.shape[1])
  output_train = output_train.reshape(output_train.shape[0],1,output_train.shape[1])
  #output_train = output_train.reshape(output_train.shape[0],1,1)

  input_test = input_test.reshape(input_test.shape[0],1,input_test.shape[1])
  output_test = output_test.reshape(output_test.shape[0],1,output_test.shape[1])
  #output_test = output_test.reshape(output_test.shape[0],1,1)


  # load Model
  print('load model...')

  model = load_model(models[i], custom_objects={'disag_error':disag_error})

  model.summary()

  mse, mae, disag = model.evaluate(input_test, output_test,
                              batch_size=batch_size)
  print('Test mse:', mse)
  print('Test mae:', mae)
  print('Test disag:', disag)

  target = model.predict(input_test)

  accs[i] = accuracy(target.reshape(num_testsamples,num_outputs),output_test.reshape(num_testsamples,num_outputs))
  disags[i] = disag


print(">>>>>Accuracy>>>>>>>")
for i,acc in enumerate(accs):
  print(datasets[i],acc)

print(">>>>>Disag>>>>>>>")
for i,disag in enumerate(disags):
  print(datasets[i],disag)



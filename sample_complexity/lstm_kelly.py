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
from keras.callbacks import EarlyStopping
import gc
from joblib import Parallel, delayed
import multiprocessing

def disag_error(y_true,y_pred):
    return K.sum(K.square(K.abs(y_pred - y_true)), axis = -1) / 2

batch_size = 150
epochs = 50
window_size = 60

print('Loading data...')
with open('../../dataset/window_stride/input_signal60.csv', 'rb') as f:
  input_signal = np.loadtxt(f, delimiter='\t')
  f.close()

print('Loading data...')
with open('../../dataset/window_stride/output_signal60_kitchen_fridge.csv', 'rb') as f:
  output_signal = np.loadtxt(f, delimiter='\t')
  f.close()

for i in range(80):
  # split data in train and test data
  input_train = input_signal[0:((input_signal.shape[0]//100)*(i+1)),:]
  output_train = output_signal[0:((output_signal.shape[0]//100)*(i+1)),:]
 
  # reshape data for lstm network
  input_train = input_train.reshape(input_train.shape[0],1,input_train.shape[1])
  output_train = output_train.reshape(output_train.shape[0],1,output_train.shape[1])

  
  # Build Model
  print('Build model {}'.format(i))
  model = Sequential()
  model.add(Conv1D(filters=16,kernel_size=4,padding='same', input_shape=(1,window_size)))
  model.add(Bidirectional(LSTM(128, activation='sigmoid', return_sequences=True))) #activation default is tanh different from recurrent_activation
  model.add(Bidirectional(LSTM(256, activation='sigmoid', return_sequences=True)))
  model.add(Dense(128,activation='tanh'))
  model.add(Dense(window_size))

  # try using different optimizers and different optimizer configs
  # Compile Model
  print('Compile model...')
  model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['mean_absolute_error', disag_error])
  # Train model ...
  early_stopping = EarlyStopping(monitor='val_loss', patience=2)
  print('Train...')
  model.fit(input_train, output_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[early_stopping])

  model.save('lstmmodel/lstmk{}.h5'.format(i))
  model = None
  gc.collect()
  K.reset_uids()
  K.clear_session()






  
  
  
  



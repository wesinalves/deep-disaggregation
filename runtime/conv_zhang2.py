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
from keras.layers import Conv1D, Dense
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping

appliances_name = ['north_bedroom', 'mastersouth_bedroom', 'basement_plugsligths', 'clothes_dryer','clothes_washer',
        'dining_roomplugs', 'dishwasher','eletronic_workbench', 'security_equipament', 'kitchen_fridge',
        'hvac', 'garage', 'heat_pump', 'hot_water', 'home_office','outside_plug','entertainment',
        'utility_plug','wall_oven']


def disag_error(y_true,y_pred):
  return K.sum(K.square(K.abs(y_pred - y_true)), axis = -1) / 2

batch_size = 150
num_appliances = 19
epochs = 50
window_size = 60
appliance = 'kettle'

print('Loading data...')
with open('../../dataset/window_stride/cost/input_signal60.csv', 'rb') as f:
  input_signal = np.loadtxt(f, delimiter='\t')
  f.close()

for i,appliance in enumerate(appliances_name):
  with open('../../dataset/window_stride/cost/output_signal_'+appliance+'.csv', 'rb') as f:
    output_signal = np.loadtxt(f, delimiter='\t')
    f.close()

  # split data in train and test data
  input_train = input_signal[0:((input_signal.shape[0]//3)*2),:]
  output_train = output_signal[0:((output_signal.shape[0]//3)*2)]

  input_test = input_signal[((input_signal.shape[0]//3)*2):input_signal.shape[0],:]
  output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0]]

  # reshape data for convolutional network
  input_train = input_train.reshape(input_train.shape[0],1,input_train.shape[1])
  output_train = output_train.reshape(output_train.shape[0],1,1)

  input_test = input_test.reshape(input_test.shape[0],1,input_test.shape[1])
  output_test = output_test.reshape(output_test.shape[0],1,1)


  # Build Model
  print('Build model...')
  model = Sequential()
  model.add(Conv1D(filters=30,kernel_size=10,padding='same', activation='relu', input_shape=(1,window_size)))
  model.add(Conv1D(filters=30,kernel_size=8,padding='same', activation='relu'))
  model.add(Conv1D(filters=40,kernel_size=6,padding='same', activation='relu'))
  model.add(Conv1D(filters=50,kernel_size=5,padding='same', activation='relu'))
  model.add(Conv1D(filters=50,kernel_size=5,padding='same', activation='relu'))
  model.add(Dense(1024,activation='relu'))
  #model.add(GlobalAveragePooling1D())
  model.add(Dense(1))


  # try using different optimizers and different optimizer configs
  # Compile Model
  print('Compile model...')
  model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_absolute_error', disag_error])
  # Train model ...
  print('Train...')
  early_stopping = EarlyStopping(monitor='val_loss', patience = 2)
  model.fit(input_train, output_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split = 0.2,
            callbacks = [early_stopping]

            )

  model.save('conv_zhang2_'+appliance+'.h5')

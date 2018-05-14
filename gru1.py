'''Trains an GRU model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers import GRU
import numpy as np

max_features = 2000
batch_size = 115

print('Loading data...')
with open('../dataset/appliances/aggregate_signal10.csv', 'rb') as f:
	aggregate_signal = np.loadtxt(f, delimiter='\t')
	f.close()

with open('../dataset/appliances/heat_pump_10.csv', 'rb') as f:
	appliance_signal = np.loadtxt(f, delimiter='\t')
	f.close()


n_dim = 10
n_sample = 1440
attribute = 5 # power
time_step = aggregate_signal.shape[0]//batch_size

#aggregate_signal = aggregate_signal.reshape(aggregate_signal.shape[0],1,aggregate_signal.shape[1])
#appliance_signal = appliance_signal.reshape(appliance_signal.shape[0],1,appliance_signal.shape[1])

print('**** shape of train and test signal ****')



print('Build model...')
model = Sequential()
#model.add(Embedding(max_features, 32))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(10,1)))
model.add(Dense(10, activation='sigmoid'))

model.summary()

# try using different optimizers and different optimizer configs
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(np.expand_dims(aggregate_signal, axis=2), appliance_signal,
          batch_size=batch_size,
          epochs=5,
          validation_data=(np.expand_dims(aggregate_signal, axis=2), appliance_signal))
score, acc = model.evaluate(np.expand_dims(aggregate_signal, axis=2), appliance_signal,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
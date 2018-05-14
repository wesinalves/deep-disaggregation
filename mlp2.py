'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

batch_size = 50
num_appliances = 4
epochs = 5

print('Loading data...')
with open('../dataset/multivariate/input_signal01.csv', 'rb') as f:
	input_signal = np.loadtxt(f, delimiter='\t')
	f.close()

with open('../dataset/multivariate/output_signal01.csv', 'rb') as f:
	output_signal = np.loadtxt(f, delimiter='\t')
	f.close()


# split data in train and test data
input_train = input_signal[0:((input_signal.shape[0]//3)*2),:]
output_train = output_signal[0:((output_signal.shape[0]//3)*2),:]

input_test = input_signal[((input_signal.shape[0]//3)*2):input_signal.shape[0],:]
output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0],:]


model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(2,)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_appliances, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

history = model.fit(input_signal, output_signal,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(input_signal, output_signal))
score = model.evaluate(input_signal, output_signal, verbose=0)

# model save requires h5py
# model.save('../dataset/multivariate/models/mlp2.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])
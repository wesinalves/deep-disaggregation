'''
This code investigate the use of Autoencoders to improve nmf results.
Autoencoders is build to reduce inputs dimensionality that will be applied into a nmf conv_autoencoder.
After that, W and H will be used to source separetion using Wi and Hi in order to disaggregate power consuption appliance.
The core idea is with Wi and Hi obtained from encoded space, we can reconstruct Pi in decoded space using decoder part from Autoencoder learnt.
Author: Wesin Alves
Data: 2017-12-22
'''
from sklearn.decomposition import NMF
from sklearn.decomposition.nmf import _beta_divergence 
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time
import sklearn.decomposition as decomp
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout
from keras.models import Model, Sequential
start_time = time.time()



###########
# load data
###########

with open('../dataset/train_signal92.csv', 'rb') as f:
	train_signal = np.loadtxt(f, delimiter='\t')
	f.close()

with open('../dataset/test_signalplus92.csv', 'rb') as f:
	test_signalplus = np.loadtxt(f, delimiter='\t')
	f.close()

n_day = 92
n_sample = 1440
attribute = 5 # power


print('**** shape of aggregate and test signal ****')
print(train_signal.shape)
print(test_signalplus.shape)

#############################
# Disaggregate error
# sum(l2_norm(Xi-Xi')**2 / 2)
#############################

def disag_error(X,W,H):
	error = (la.norm((X-W.dot(H)),2)**2)/2
	return error

##################################
# build standart autoencoder conv_autoencoder
##################################

#this is a size of encoded representation
encoding_dim = 16
input_dim = n_day

# this conv_autoencoder maps an input to its reconstruction
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(input_dim,1)))
conv_autoencoder.add(MaxPooling1D(2))
conv_autoencoder.add(Conv1D(19, 3, activation='relu'))
conv_autoencoder.add(MaxPooling1D(2))
conv_autoencoder.add(Conv1D(19, 3, activation='relu'))
conv_autoencoder.add(UpSampling1D(2))
conv_autoencoder.add(Conv1D(64, 3, activation='relu'))
conv_autoencoder.add(UpSampling1D(2))

conv_autoencoder.add(Dropout(0.5))
conv_autoencoder.add(Flatten())
conv_autoencoder.add(Dense(input_dim, activation='sigmoid'))


#train the conv_autoencoder
conv_autoencoder.compile(optimizer='adam', 
						loss='mean_squared_error',
						metrics=['accuracy'])

conv_autoencoder.summary()

###################
# train autoencoder
###################


conv_autoencoder.fit(np.expand_dims(train_signal, axis=2),train_signal,epochs=300,
	batch_size=64,
	shuffle=True,
	validation_data=(np.expand_dims(test_signalplus, axis=2),test_signalplus))

######################
# evaluate autoencoder
######################

score = conv_autoencoder.evaluate(np.expand_dims(test_signalplus, axis=2), test_signalplus, verbose=0)
print('*'*5,'evaluate conv_autoencoder','*'*5)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
################
# encoder inputs
################

test_encoded = encoder.predict(test_signal)
train_encoded = encoder.predict(aggregate_signal)


################
# buid nmf conv_autoencoder
################

alpha = 0.012
conv_autoencoder = NMF(n_components = encoding_dim, init = 'random', max_iter=500, solver='cd')

#################
# train nmf conv_autoencoder
#################
print('*'*5,'evaluate NMF','*'*5)
W = conv_autoencoder.fit_transform(train_encoded)
train_error = disag_error(train_encoded,W,conv_autoencoder.components_)
print('train error: ', train_error)

####################
# evaluate nmf conv_autoencoder
####################
W_test = conv_autoencoder.transform(test_encoded)
test_error = disag_error(test_encoded,W_test,conv_autoencoder.components_)
print('Test error: ', test_error)

#################
# decoder outputs
#################

decoded_signal = decoder.predict(W.dot(conv_autoencoder.components_))

##############
# get accuracy
##############

print('*'*5,'evaluate nmf conv_autoencoder','*'*5)
score = conv_autoencoder.evaluate(decoded_signal, test_signal, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
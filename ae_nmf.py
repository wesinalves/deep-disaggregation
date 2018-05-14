'''
This code investigate the use of Autoencoders to improve nmf results.
Autoencoders is build to reduce inputs dimensionality that will be applied into a nmf model.
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
start_time = time.time()
import sklearn.decomposition as decomp
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD,RMSprop,Adam,Adadelta,Adagrad
from keras.regularizers import l2,l1

###########
# load data
###########

with open('../dataset/train_signal10.csv', 'rb') as f:
	train_signal = np.loadtxt(f, delimiter='\t')
	f.close()

with open('../dataset/test_signalplus10.csv', 'rb') as f:
	test_signalplus = np.loadtxt(f, delimiter='\t')
	f.close()

n_day = 10
n_sample = 1440
attribute = 5 # power


print('**** shape of train and test signal ****')

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
# build standart autoencoder model
##################################

#this is a size of encoded representation
encoding_dim = 19
input_dim = n_day

#this is our input placeholder
input_ = Input(shape=(input_dim,))
#encoded is a encoded representation of the input
encoded = Dense(input_dim, activation='relu')(input_)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
#decoded is the lossy reconstruction of the input
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='relu')(decoded)

# this model maps an input to its reconstruction
deep_autoencoder = Model(input_, decoded)
#this model maps an input to  its encoded representation - outputs 32 dmin
encoder = Model(input_, encoded)

#create a placeholder for a encoded (32-dimension) input
encoded_input = Input(shape=(encoding_dim,))
#retrieve the last layer of the deep_autoencoder model - output 64 dim, after 128, after 784
decoder_layer1 = deep_autoencoder.layers[-4]
decoder_layer2 = deep_autoencoder.layers[-3]
decoder_layer3 = deep_autoencoder.layers[-2]
decoder_layer4 = deep_autoencoder.layers[-1]

decoder_layer = decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

#create the  decoder model
decoder = Model(encoded_input, decoder_layer)

#train the deep_autoencoder
deep_autoencoder.compile(optimizer='adam', 
						loss='mean_squared_error',
						metrics=['accuracy'])

deep_autoencoder.summary()

###################
# train autoencoder
###################

deep_autoencoder.fit(train_signal,train_signal,epochs=100,
	batch_size=64,
	shuffle=True,
	validation_data=(test_signalplus,test_signalplus))

######################
# evaluate autoencoder
######################

score = deep_autoencoder.evaluate(test_signalplus, test_signalplus, verbose=0)
print('*'*5,'evaluate deep_autoencoder','*'*5)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

################
# encoder inputs
################

'''
test_encoded = encoder.predict(test_signal)
train_encoded = encoder.predict(aggregate_signal)


################
# buid nmf model
################

alpha = 0.012
model = NMF(n_components = encoding_dim, init = 'random', max_iter=500, solver='cd')

#################
# train nmf model
#################
print('*'*5,'evaluate NMF','*'*5)
W = model.fit_transform(train_encoded)
train_error = disag_error(train_encoded,W,model.components_)
print('train error: ', train_error)

####################
# evaluate nmf model
####################
W_test = model.transform(test_encoded)
test_error = disag_error(test_encoded,W_test,model.components_)
print('Test error: ', test_error)

#################
# decoder outputs
#################

decoded_signal = decoder.predict(W.dot(model.components_))

##############
# get accuracy
##############

print('*'*5,'evaluate nmf deep_autoencoder','*'*5)
score = deep_autoencoder.evaluate(decoded_signal, test_signal, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
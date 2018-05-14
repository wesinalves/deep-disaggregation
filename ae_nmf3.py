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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


###########
# load data
###########

with open('../dataset/aggregate_signal92.csv', 'rb') as f:
	aggregate_signal = np.loadtxt(f, delimiter='\t')
	f.close()

with open('../dataset/test_signal92.csv', 'rb') as f:
	test_signal = np.loadtxt(f, delimiter='\t')
	f.close()

n_day = 92
n_sample = 1440
attribute = 5 # power

train_signal = np.zeros((n_sample*20,n_day))
for k in range(20):
	train_signal[k*aggregate_signal.shape[0]:aggregate_signal.shape[0] + aggregate_signal.shape[0]*k,:] = aggregate_signal

test_signalplus = np.zeros((n_sample*4,n_day))
for k in range(4):
	test_signalplus[k*test_signal.shape[0]:test_signal.shape[0] + test_signal.shape[0]*k,:] = test_signal

train_signal = train_signal.reshape(20,n_sample,train_signal.shape[1],1)
test_signalplus = test_signalplus.reshape(4,n_sample,test_signalplus.shape[1],1)




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
encoding_dim = 16
input_dim = n_day

#this is our input placeholder
input_ = Input(shape=(n_sample,input_dim,1))
#encoded is a encoded representation of the input
x = Conv2D(64, (3,3), activation='relu', padding='same')(input_)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(encoding_dim, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

#decoded is the lossy reconstruction of the input
x = Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32,(3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3), activation='sigmoid', padding='same')(x)

# this model maps an input to its reconstruction
conv_autoencoder = Model(input_, decoded)


#train the conv_autoencoder
conv_autoencoder.compile(optimizer='adadelta', 
						loss='mean_squared_error',
						metrics=['accuracy'])

conv_autoencoder.summary()

#this model maps an input to  its encoded representation - outputs 32 dmin
encoder = Model(input_, encoded)

#create a placeholder for a encoded (32-dimension) input
encoded_input = Input(shape=(1,n_sample,encoding_dim))
#retrieve the last layer of the conv_autoencoder model - output 64 dim, after 128, after 784
decoder_layer1 = conv_autoencoder.layers[-5]
decoder_layer2 = conv_autoencoder.layers[-4]
decoder_layer3 = conv_autoencoder.layers[-3]
decoder_layer4 = conv_autoencoder.layers[-2]
decoder_layer5 = conv_autoencoder.layers[-1]

decoder_layer = decoder_layer5(decoder_layer4(
					decoder_layer3(decoder_layer2(
						decoder_layer1(encoded_input)))))

#create the  decoder model
decoder = Model(encoded_input, decoder_layer)

###################
# train autoencoder
###################


conv_autoencoder.fit(train_signal,train_signal,epochs=50,
	batch_size=16,
	shuffle=True,
	validation_data=(test_signalplus,test_signalplus))

######################
# evaluate autoencoder
######################

score = conv_autoencoder.evaluate(test_signalplus, test_signalplus, verbose=0)
print('*'*5,'evaluate conv_autoencoder','*'*5)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

################
# encoder inputs
################

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

print('*'*5,'evaluate nmf conv_autoencoder','*'*5)
score = conv_autoencoder.evaluate(decoded_signal, test_signal, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

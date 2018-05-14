'''
This code investigate the use of Variational Autoencoders (VAE) to improve nmf results.
VAE is build to reduce inputs dimensionality that will be applied into a nmf model.
After that, W and H will be used to source separetion using Wi and Hi in order to disaggregate power consuption appliance.
The core idea is with Wi and Hi obtained from latent space, we can reconstruct Pi in real space using decoder part from Autoencoder learnt.
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
from keras.layers import Input, Dense, Dropout, Lambda, Layer
from keras.models import Model
from keras.optimizers import SGD,RMSprop,Adam,Adadelta,Adagrad
from keras.regularizers import l2,l1
from keras import backend as K
from keras import metrics


start_time = time.time()


###########
# load data
###########

with open('../dataset/train_signal90.csv', 'rb') as f:
	train_signal = np.loadtxt(f, delimiter='\t')
	f.close()

with open('../dataset/test_signalplus90.csv', 'rb') as f:
	test_signalplus = np.loadtxt(f, delimiter='\t')
	f.close()

n_day = 90
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
##################
batch_size = 100
original_dim = n_day
latent_dim = 19
intermediate_dim = 32
epochs = 100
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# function to generate sampling based on data distribution
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling)([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Custom loss layer

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)

    return recon + kl

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=['accuracy'])
vae.summary()

###################
# train autoencoder
###################

vae.fit(train_signal,train_signal, epochs=epochs,
	batch_size=batch_size,
	shuffle=True,
	validation_data=(test_signalplus,test_signalplus))

######################
# evaluate autoencoder
######################

score = vae.evaluate(test_signalplus, test_signalplus, verbose=0)
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
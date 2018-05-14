from __future__ import print_function

from keras.models import Sequential
from keras.layers import LSTM,Conv1D, Bidirectional, Dense, Flatten
import numpy as np
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot  as plt

def disag_error(y_true,y_pred):
  return K.sum(K.square(K.abs(y_pred - y_true)), axis = -1) / 2

print('Loading data...')
with open('../dataset/multivariate/input_dolly.csv', 'rb') as f:
  input_mult = np.loadtxt(f, delimiter='\t')
  f.close()
with open('../dataset/multivariate/output_dolly.csv', 'rb') as f:
  output_mult = np.loadtxt(f, delimiter='\t')
  f.close()

with open('../dataset/window_stride/input_dolly.csv', 'rb') as f:
	input_seq = np.loadtxt(f, delimiter='\t')
	f.close()
with open('../dataset/window_stride/output_dolly.csv', 'rb') as f:
	output_seq = np.loadtxt(f, delimiter='\t')
	f.close()

'''
input_seq = input_signal[((input_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),0]
output_seq = output_signal[((input_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),2]

input_mult = input_signal[((input_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),:]
output_mult = output_signal[((output_signal.shape[0]//100)*3):((input_signal.shape[0]//100)*4),:]
'''


timesteps = 60

input_seq = input_seq.reshape(input_seq.shape[0]//timesteps,1,timesteps)
output_seq = output_seq.reshape(output_seq.shape[0]//timesteps,1,timesteps)
num_features = input_mult.shape[1]
num_outputs = output_mult.shape[1]
input_mult = input_mult.reshape(input_mult.shape[0]//timesteps,timesteps,num_features)
output_mult = output_mult.reshape(output_mult.shape[0]//timesteps,timesteps,num_outputs)


model_conv = load_model('conv_zhang1e.h5', custom_objects={'disag_error':disag_error})
target_conv = model_conv.predict(input_seq)
model_conv = None
K.reset_uids()
K.clear_session()
model_lstm = load_model('lstm_kelly5_withstride.h5', custom_objects={'disag_error':disag_error})
target_lstm = model_lstm.predict(input_seq)
model_lstm = None
K.reset_uids()
K.clear_session()

model_gru = load_model('gru15.h5', custom_objects={'disag_error':disag_error})
target_gru = model_gru.predict(input_mult)
model_gru = None
K.reset_uids()
K.clear_session()


y_predconv = target_conv[17][0]
y_predlstm = target_lstm[17][0]
y_predgru = target_gru[17][:,2]
y = output_seq[17][0]

opacity = 0.4

plt.plot(y_predconv, 'bs-', label='convz', alpha=opacity)
plt.plot(y_predlstm, 'r--', label='lstmk', alpha=opacity)
plt.plot(y_predgru, 'yo-', label='proposed', alpha=opacity)
plt.plot(y, color='k', label='ground_truth', alpha=opacity)
plt.xlabel("Samples")
plt.ylabel("Fridge")
plt.legend()
plt.show()

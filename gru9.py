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
from keras.layers import GRU,LSTM
import numpy as np
import keras.backend as K

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

batch_size = 150
num_appliances = 19
epochs = 5



print('Loading data...')
with open('../dataset/multivariate/input_signal04.csv', 'rb') as f:
  input_signal = np.loadtxt(f, delimiter='\t')
  f.close()

with open('../dataset/multivariate/output_signal04.csv', 'rb') as f:
  output_signal = np.loadtxt(f, delimiter='\t')
  f.close()

# split data in train and test data
input_train = input_signal[0:((input_signal.shape[0]//3)*2),:]
output_train = output_signal[0:((output_signal.shape[0]//3)*2),:]

input_test = input_signal[((input_signal.shape[0]//3)*2):input_signal.shape[0],:]
output_test = output_signal[((output_signal.shape[0]//3)*2):output_signal.shape[0],:]

# reshape data for lstm network
input_train = input_train.reshape(input_train.shape[0],1,input_train.shape[1])
output_train = output_train.reshape(output_train.shape[0],1,output_train.shape[1])

input_test = input_test.reshape(input_test.shape[0],1,input_test.shape[1])
output_test = output_test.reshape(output_test.shape[0],1,output_test.shape[1])

print('**** shape of train signal ****')
print(input_train.shape)
print(output_train.shape)
print('**** shape of test signal ****')
print(input_test.shape)
print(output_test.shape)

# Build Model
print('Build model...')
model = Sequential()
model.add(GRU(num_appliances, dropout=0.2, recurrent_dropout=0.2, input_shape=(1,4), return_sequences=True, activation='sigmoid' ))


model.summary()

# try using different optimizers and different optimizer configs
# Compile Model
print('Compile model...')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy', fmeasure, recall, precision])
# Train model ...
print('Train...')
model.fit(input_train, output_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(input_test, output_test),
          verbose=0)

score, acc, f1, recall, precision = model.evaluate(input_test, output_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)
print('Test f1:', f1)
print('Test recall:', recall)
print('Test precision:', precision)


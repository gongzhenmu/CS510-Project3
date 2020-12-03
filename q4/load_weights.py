import pickle
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_curve, auc
from text_rnn import TextRNN, VariedLayerSizeTextRNN, MultiLayerTextRNN
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

with open('../data/x_train_full.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('../data/vocab_set_full.pickle', 'rb') as handle:
    vocabulary_set = pickle.load(handle)
with open('../data/x_test_full.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open('../data/y_test_full.pickle', 'rb') as handle:
    Y_test = pickle.load(handle)
    
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
maxlen = 1000
max_features = encoder.vocab_size
embedding_dims = 64
epochs = 1
learning_rate = 0.0001
loss_fn = 'binary_crossentropy'
batch_size = 256
model_name = 'TextRNN'
layer_size = 128

model = TextRNN(maxlen,max_features,embedding_dims)
# model = MultiLayerTextRNN(maxlen,max_features,embedding_dims, layer_size)
model.build(X_train.shape)

# model.load_weights('../data/models/model-TextRNN-01-0.66301.hdf5')
model.load_weights('../data/models/model-TextRNN-01-0.48146.hdf5')

# Building generators
class CustomGenerator(Sequence):
    def __init__(self, text, labels, batch_size, num_steps=None):
        self.text, self.labels = text, labels
        self.batch_size = batch_size
        self.len = np.ceil(len(self.text) / float(self.batch_size)).astype(np.int64)
        if num_steps:
            self.len = min(num_steps, self.len)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        batch_x = self.text[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

test_gen = CustomGenerator(X_test, Y_test, batch_size)
predIdxs = model.predict_generator(test_gen, verbose=2)

fpr, tpr, _ = roc_curve(Y_test, predIdxs)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig('auc_model_' + str(model_name) + '_Full_' + str(epochs) + '_' + str(learning_rate) + '_' + str(batch_size) + '_' + str(loss_fn) + '.png')

import pickle
import pandas as pd
import numpy as np
import sys
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


with open('../data/y_train.pickle', 'rb') as handle:
    Y_train = pickle.load(handle)
with open('../data/y_test.pickle', 'rb') as handle:
    Y_test = pickle.load(handle)
with open('../data/y_valid.pickle', 'rb') as handle:
    Y_valid = pickle.load(handle)

with open('../data/x_train.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('../data/x_test.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open('../data/x_valid.pickle', 'rb') as handle:
    X_valid = pickle.load(handle)
with open('../data/vocab_set.pickle', 'rb') as handle:
    vocabulary_set = pickle.load(handle)

# print("Q3: Training TextRNN on full dataset")
X_train = X_train[:50000]
Y_train = Y_train[:50000]
X_test = X_test[:25000]
Y_test = Y_test[:25000]
X_valid = X_valid[:25000]
Y_valid = Y_valid[:25000]

# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
maxlen = 1000
max_features = encoder.vocab_size
loss_fn = 'binary_crossentropy'
embedding_dims = 64

# epochs = 10
# learning_rate = 1e-4
# batch_size = 256
# model_name = 'TextAttBiRNN_Two_Layers'
# layer_size=128

# Read stdin: epochs, learning_rate, batch_size, model_name layer_size
epochs = int(sys.argv[1])
learning_rate = float(sys.argv[2])
batch_size = int(sys.argv[3])
model_name = sys.argv[4]
layer_size = sys.argv[5]

# Model Definition
if model_name == 'TextRNN':
    model = TextRNN(maxlen,max_features,embedding_dims)
elif model_name == 'VariedLayerSizeTextRNN':
    model = VariedLayerSizeTextRNN(maxlen,max_features,embedding_dims, layer_size)
elif model_name == 'MultiLayerTextRNN':
    model = MultiLayerTextRNN(maxlen,max_features,embedding_dims, layer_size)
else:
    print("Invalid model name, exiting")
    exit()
    
# Learning rate
opt = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

print("Training {0} with {1} epochs, {2} learning rate, {3} batch size, and {4} layer_size".format(
    model_name,
    epochs,
    learning_rate,
    batch_size,
    layer_size))

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


train_gen = CustomGenerator(X_train, Y_train, batch_size)
valid_gen = CustomGenerator(X_valid, Y_valid, batch_size)
test_gen = CustomGenerator(X_test, Y_test, batch_size)

# Training the model
checkpointer = ModelCheckpoint('../data/models/model-' + str(model_name) + '-{epoch:02d}-{val_loss:.5f}.hdf5',
                               monitor='val_loss',
                               verbose=2,
                               save_weights_only=False,
                               save_best_only=True,
                               mode='min')

callback_list = [checkpointer] #, , reduce_lr

his1 = model.fit_generator(
                    generator=train_gen,
                    epochs=epochs,
                    validation_data=valid_gen,
                    callbacks=callback_list)

# model.summary()
                                                       
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

plt.savefig('roc_curves/auc_model_' + str(model_name) + '_' + str(epochs) + '_' + str(learning_rate) + '_' + str(batch_size) + '_' + str(loss_fn) + '.png')

                    
                    
                    
                    
                    

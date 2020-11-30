import pickle
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 1000

# Loading tokenized data
with open('../data/tokenized_train.pickle', 'rb') as handle:
    train = pickle.load(handle)
with open('../data/tokenized_valid.pickle', 'rb') as handle:
   valid = pickle.load(handle)
with open('../data/tokenized_test.pickle', 'rb') as handle:
   test = pickle.load(handle)

# Reshape instances:
def reshape_instances(df):
    df["input"] = df["context_before"].apply(lambda x: " ".join(x)) + " <START> " + df["instance"].apply(lambda x: " ".join(x)) + " <END> " + df["context_after"].apply(lambda x: " ".join(x))
    X_df = []
    Y_df = []
    for index, rows in df.iterrows():
        X_df.append(rows.input)
        Y_df.append(rows.is_buggy)
    return X_df, Y_df

print("Reshaping instances")
X_train, Y_train = reshape_instances(train)
X_test, Y_test = reshape_instances(test)
X_valid, Y_valid = reshape_instances(valid)

# Use a subset of data to save time
# You can change it in Part(III) to improve your result
print("Preprocessing full dataset")

# Build vocabulary and encoder from the training instances
vocabulary_set = set()
for data in X_train:
   vocabulary_set.update(data.split())

print("Encoding")
# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

def encode(text):
  encoded_text = encoder.encode(text)
  return encoded_text

X_train = list(map(lambda x: encode(x), X_train))
X_test = list(map(lambda x: encode(x), X_test))
X_valid = list(map(lambda x: encode(x), X_valid))

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
X_valid = pad_sequences(X_valid, maxlen=maxlen)

with open('../data/y_train_full.pickle', 'wb') as handle:
    pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/y_test_full.pickle', 'wb') as handle:
    pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/y_valid_full.pickle', 'wb') as handle:
    pickle.dump(Y_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/x_train_full.pickle', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/x_test_full.pickle', 'wb') as handle:
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/x_valid_full.pickle', 'wb') as handle:
    pickle.dump(X_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/vocab_set_full.pickle', 'wb') as handle:
    pickle.dump(vocabulary_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
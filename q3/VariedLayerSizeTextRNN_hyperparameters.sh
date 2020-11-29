#!/bin/bash

# python train_and_test.py epochs, learning_rate, batch_size, model_name layer_size

# Vary learning rate + layer size
python train_and_test.py 20 0.001 256 VariedLayerSizeTextRNN 128
python train_and_test.py 20 0.0001 256 VariedLayerSizeTextRNN 128
python train_and_test.py 20 0.00001 256 VariedLayerSizeTextRNN 128
python train_and_test.py 20 0.001 256 VariedLayerSizeTextRNN 256
python train_and_test.py 20 0.0001 256 VariedLayerSizeTextRNN 256
python train_and_test.py 20 0.00001 256 VariedLayerSizeTextRNN 256
python train_and_test.py 20 0.001 256 VariedLayerSizeTextRNN 512
python train_and_test.py 20 0.0001 256 VariedLayerSizeTextRNN 512
python train_and_test.py 20 0.00001 256 VariedLayerSizeTextRNN 512
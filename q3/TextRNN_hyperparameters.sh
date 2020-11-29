#!/bin/bash

# python train_and_test.py epochs, learning_rate, batch_size, model_name layer_size

# Vary learning rate
python train_and_test.py 20 0.01 256 TextRNN 128
python train_and_test.py 20 0.001 256 TextRNN 128
python train_and_test.py 20 0.0001 256 TextRNN 128
python train_and_test.py 20 0.00001 256 TextRNN 128

# Varying batch size
python train_and_test.py 5 0.001 128 TextRNN 128
python train_and_test.py 5 0.001 256 TextRNN 128
python train_and_test.py 5 0.001 512 TextRNN 128
#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
from tensorflow.keras import backend as K

inputs = tf.keras.Input(shape=(1, 50))
x = tf.keras.layers.LSTM(80, return_sequences=True)(inputs)
#x = tf.keras.layers.LSTM(80, return_sequences=True)(x)
#x = tf.keras.layers.LSTM(40, return_sequences=True)(x)
x = tf.keras.layers.LSTM(30, return_sequences=True)(x)

outputs = tf.keras.layers.Dense(20)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.save("model.h5")

model.compile(loss='mse')
model.summary()


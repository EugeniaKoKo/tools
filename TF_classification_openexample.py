try:
  # Colab only
  %tensorflow_version 2.x
except Exception:
    pass


from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt


# load dataset
print("load dataset...")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# preprocessing dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# set layers
print("create model...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train model
print("train model...")
model.fit(train_images, train_labels, epochs=10)

# evalueate acuuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nТочность на проверочных данных:', test_acc)
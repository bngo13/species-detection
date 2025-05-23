import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd
from keras.src.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import shutil
import os

from tensorflow.python.keras.utils.vis_utils import plot_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

IMG_DIR_LOCATION = "./PROJDATASET_SMALL"
DEFAULT_WEIGHTS = "./mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

def train_model(dropout, layer1, layer2):
  batch_size = 16
  img_height = 750
  img_width = 750

  train_ds = tf.keras.utils.image_dataset_from_directory(
    IMG_DIR_LOCATION,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    IMG_DIR_LOCATION,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  class_names = train_ds.class_names
  print(class_names)

  for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  normalization_layer = layers.Rescaling(1./255)

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  first_image = image_batch[0]
  # Notice the pixel values are now in `[0,1]`.
  print(np.min(first_image), np.max(first_image))

  num_classes = len(class_names)

  data_augmentation = keras.Sequential([
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.2),
      layers.RandomZoom(0.2),
      layers.RandomContrast(0.2)
  ])

  local_weights_path = pathlib.Path(DEFAULT_WEIGHTS)
  base_model = tf.keras.applications.MobileNetV2(input_shape=(750, 750, 3),include_top=False,weights=None)
  base_model.load_weights(local_weights_path)
  base_model.trainable = False  # Freeze the convolutional base

  inputs = keras.Input(shape=(750, 750, 3))
  x = data_augmentation(inputs)  # Augmentation layer if desired
  x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
  x = base_model(x, training=False)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(layer1, activation='relu')(x)
  x = layers.Dense(layer2, activation='relu')(x)
  outputs = layers.Dense(num_classes, activation="softmax")(x)
  model = keras.Model(inputs, outputs)

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.summary()

  early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    mode='max'  # Use 'max' for accuracy
  )

  epochs=1000
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping_callback]
  )

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)
  predictions = model.predict(val_ds)

  evaluate = model.evaluate(val_ds)

  return model, max(val_acc), [acc, val_acc, loss, val_loss, epochs_range]

def training_all():
  dropouts = [0.3, 0.5, 0.7]
  layer1s = [512, 1024, 2048]
  layer2s = [512, 1024, 2048]
  results = []
  for dropout in dropouts:
    for layer1 in layer1s:
      for layer2 in layer2s:
        model, score, stats = train_model(dropout, layer1, layer2)
        print(score)
        results.append([score*100, [dropout, layer1, layer2], stats])

  results.sort(key=lambda x: x[1], reverse=True)
  print(max(results))
  for i in range(10):
    best_result = results[i-1]
    print(best_result[0])
    print(f"Dropout: {best_result[1][0]}")
    print(f"Layer 1: {best_result[1][1]}")
    print(f"Layer 2: {best_result[1][2]}")

training_all()

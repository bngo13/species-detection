import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd
import cv2
from keras.src.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import shutil
import os

from tensorflow.python.keras.utils.vis_utils import plot_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
"""dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')"""
DIR_LOCATION = "../../PROJDATASET"
IMG_DIR_LOCATION = "../../TENSOR_PROJDATASET_copy"
ID_LIST = pd.read_csv(f"{DIR_LOCATION}/ostracod_IDs.csv")


def load_dataset():
  # Initialize Image DataFrame
  data_df = pd.DataFrame()

  # Load Image Data to DataFrame
  for _, row in ID_LIST.iterrows():
    #print(f"Reading Index: {row['OrganismID']}")

    # Load the image into the row
    dir_path = list(pathlib.Path(DIR_LOCATION).glob(f"{row['OrganismID']:03}*"))[0]
    file_paths = pathlib.Path(dir_path).glob("*.tif")
    for file in file_paths:
      headers = np.array([row["Species"], row["Sex"], row["Stage"], row["location"], str(file)])
      header_df = pd.DataFrame(headers).T
      header_df[2] = header_df[2] if header_df[2].isna else header_df[2].astype(np.uint8)
      row_df = header_df
      data_df = pd.concat([data_df, row_df])
  data_df = data_df.reset_index()
  data_df = data_df.drop("index", axis=1)
  data_df = data_df.dropna()

  # Rename columns
  data_df.columns.values[0] = "species"
  data_df.columns.values[1] = "sex"
  data_df.columns.values[2] = "stage"
  data_df.columns.values[3] = "location"
  data_df.columns.values[4] = "filepath"

  return data_df

def image_dataset(data_df):
  return pd.Series.tolist(data_df.iloc[:,[0,4]])

def train_model(image_df, dropout, layer1, layer2):
  #image_df = image_dataset()
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

  """plt.figure(figsize=(10, 10))
  for images, labels in train_ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")"""

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

  local_weights_path = pathlib.Path(f"mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
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
  """model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
    layers.Dense(num_classes)
  ])
  
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])"""

  model.summary()
  checkpoint_callback = ModelCheckpoint(
    filepath=f'../../Models/model_acc.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max'  # Use 'max' for accuracy
  )

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
    callbacks=[checkpoint_callback, early_stopping_callback]
  )

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)
  predictions = model.predict(val_ds)
  """for i in predictions:
    score = tf.nn.softmax(i)

    print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
    )"""
  """for image in val_ds:
    print(image)
    img = image
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])

    print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
    )"""

  evaluate = model.evaluate(val_ds)
  model.save(f"../../Models/model_acc_{evaluate[1]}.keras")

  #training_plot(acc, val_acc, loss, val_loss, epochs_range)

  return model, max(val_acc), [acc, val_acc, loss, val_loss, epochs_range]


def training_plot(acc, val_acc, loss, val_loss, epochs_range):
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()


def training_all():
  dropouts = [0.3, 0.5, 0.7]
  layer1s = [512, 1024]
  layer2s = [1024, 2048]
  results = []
  data_df = image_dataset(load_dataset())
  for dropout in dropouts:
    for layer1 in layer1s:
      for layer2 in layer2s:
        model, score, stats = train_model(data_df.copy(), dropout, layer1, layer2)
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
    #training_plot(best_result[2][0], best_result[2][1], best_result[2][2], best_result[2][3], best_result[2][4])


#train_model(0,0,0,0)
training_all()
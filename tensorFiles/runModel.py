import pathlib
from tkinter.filedialog import askopenfile, askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import os

# Input Constants
IMG_DIR_LOCATION = "../../TENSOR_PROJDATASET_copy"
IMG_HEIGHT = 750
IMG_WIDTH = 750
WEIGHTS_LOCATION = "./mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
SUPPORTED_FILE_TYPES = (
    '.jpg',
    '.jpeg',
    '.png',
    '.tif'
)

# Output Constants
OUTPUT_LOCATION = "./Models"

# Model Constants
BATCH_SIZE = 16
EPOCHS = 200
K_FOLDS = 10
DROPOUT = 0.5

# Create tf.data.Dataset from file paths and labels
def create_dataset():
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

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)
    return train_ds, val_ds, class_names, num_classes

def create_model():
    local_weights_path = pathlib.Path(f"mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(750, 750, 3), include_top=False, weights=None)
    base_model.load_weights(local_weights_path)
    base_model.trainable = False

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2)
    ])

    inputs = keras.Input(shape=(750, 750, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

def build_model(num_classes, dropout, layer1, layer2):
    model = create_model()
    model.load_weights(askopenfilename())
    return model

def run_model():
    train_ds, val_ds, class_names, num_classes = create_dataset()
    model = build_model(0,0,0,0)
    evaluate = model.evaluate(val_ds)

    return evaluate

# Evaluate the model
evaluate = run_model()
print(evaluate)
#print("Restored model, accuracy: {:5.2f}%".format(100 * evaluate.acc))
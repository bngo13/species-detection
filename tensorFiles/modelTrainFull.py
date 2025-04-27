import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
import os

# Input Constants
IMG_DIR_LOCATION = "./PROJDATASET"
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
EPOCHS = 300
K_FOLDS = 10
DROPOUT = 0.5

# Helper to load image paths and labels
def get_image_paths_and_labels(base_dir):
    class_names = sorted(os.listdir(base_dir))
    filepaths = []
    labels = []

    for label_index, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(SUPPORTED_FILE_TYPES):
                    filepaths.append(os.path.join(class_dir, file))
                    labels.append(label_index)

    return np.array(filepaths), np.array(labels), class_names

# Create tf.data.Dataset from file paths and labels
def create_dataset(filepaths, labels, augment=False, shuffle=False):
    path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    def decode_img(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])  # Set static shape after decoding
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        return img

    def process_path(file_path, label):
        img = decode_img(file_path)
        return img, label

    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    if augment:
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2)
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# Build the model
def build_model(num_classes, dropout, layer1, layer2):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights=None
    )
    base_model.load_weights(WEIGHTS_LOCATION)
    base_model.trainable = False
    # for layer in base_model.layers[:50]: layer.trainable = True

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(layer1, activation='relu')(x)
    x = layers.Dense(layer2, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# K-Fold training loop
def run_kfold_training(layer1, layer2):
    filepaths, labels, class_names = get_image_paths_and_labels(IMG_DIR_LOCATION)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    folds = [1, 2, 3, 4, 5, 6, 7, 9, 10]

    for fold in folds:
        train_idx, val_idx = list(skf.split(filepaths, labels))[fold - 1]

        print(f"\n--- Fold {fold} ---")

        train_ds = create_dataset(filepaths[train_idx], labels[train_idx], augment=True, shuffle=True)
        val_ds = create_dataset(filepaths[val_idx], labels[val_idx])

        model = build_model(num_classes=len(class_names), dropout=DROPOUT, layer1=layer1, layer2=layer2)

        checkpoint_callback = ModelCheckpoint(
            filepath=f"{OUTPUT_LOCATION}/model_fold{fold}.keras",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max'  # Use 'max' for accuracy
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_accuracy',
            patience=100,
            restore_best_weights=True,
            mode='max'  # Use 'max' for accuracy
        )

        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_callback, early_stopping_callback])


# Run it
run_kfold_training(layer1=1024, layer2=1024)

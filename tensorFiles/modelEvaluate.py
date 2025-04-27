import tensorflow as tf

modelList = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
]

IMG_DIR = "./TESTDATASET/"
IMG_W = 750
IMG_H = 750

for model in modelList:
    model_dir = f"./Models/model_fold{model}.keras"

    model = tf.keras.models.load_model(model_dir)
    dataset = tf.keras.utils.image_dataset_from_directory(IMG_DIR, image_size=(IMG_H, IMG_W))

    print(model.evaluate(dataset))

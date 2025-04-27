import tensorflow as tf

MODEL = './Models/model_fold1.keras'

IMG_DIR = "./TESTDATASET/"
IMG_W = 750
IMG_H = 750

model = tf.keras.models.load_model(MODEL)
dataset = tf.keras.utils.image_dataset_from_directory(IMG_DIR, image_size=(IMG_H, IMG_W))

print(model.evaluate(dataset))

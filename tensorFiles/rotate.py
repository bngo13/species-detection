import pandas as pd
import os
import pathlib
from PIL import Image, ImageEnhance


IMG_DIR_LOCATION = "../../VALIDATION_DATASET"
DIR_LOCATION = "../../TENSOR_PROJDATASET_copy copy"
Folders = ["carcharodonta","morini","other"]

def resize_image(img, percentage):
    width, height = img.size
    new_width = int(width * (percentage / 100))
    new_height = int(height * (percentage / 100))
    resized_img = img.resize((new_width, new_height))
    return resized_img

def load_dataset():
    data_df = pd.DataFrame()
    for folder in Folders:
        load_path = pathlib.Path(f"{DIR_LOCATION}/{folder}")
        store_path = pathlib.Path(f"{IMG_DIR_LOCATION}/{folder}")
        file_paths = pathlib.Path(load_path).glob("*.png")
        print(file_paths)
        for file in file_paths:
            image = Image.open(file)
            name, extension = os.path.splitext(file)
            resized_image = resize_image(image, 80)
            rotated_image = resized_image.rotate(angle=90)
            enhancer = ImageEnhance.Contrast(rotated_image)
            contrast_factor = 1.2
            contrast_image = enhancer.enhance(contrast_factor)
            contrast_image.save(f"{name}1{extension}")
            image = Image.open(file)
            resized_image = resize_image(image, 65)
            rotated_image = resized_image.rotate(angle=180)
            enhancer = ImageEnhance.Contrast(rotated_image)
            contrast_factor = 0.8
            contrast_image = enhancer.enhance(contrast_factor)
            contrast_image.save(f"{name}2{extension}")
            image = Image.open(file)
            resized_image = resize_image(image, 70)
            rotated_image = resized_image.rotate(angle=270)
            enhancer = ImageEnhance.Contrast(rotated_image)
            contrast_factor = 1.0
            contrast_image = enhancer.enhance(contrast_factor)
            contrast_image.save(f"{name}3{extension}")
            os.remove(file)

load_dataset()
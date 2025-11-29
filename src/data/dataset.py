import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from src.config import IMAGE_SIZE

def read_metadata(csv_path):
    return pd.read_csv(csv_path)

def load_images_from_metadata(dataset, image_dir):
    images = []
    labels = []
    for _, row in dataset.iterrows():
        img_path = os.path.join(image_dir, row['image'])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=IMAGE_SIZE, color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(0 if row['rotulo'] == 'n' else 1)
    return np.array(images), np.array(labels)

def load_data(image_dir, csv_path):
    dataset = read_metadata(csv_path)
    return load_images_from_metadata(dataset, image_dir)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_data_generator():
    return ImageDataGenerator(
        rotation_range=100,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

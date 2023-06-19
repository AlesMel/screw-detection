import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image as image_utils


def preprocess_images(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image, label


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image_utils.load_img(image_path, target_size=target_size)
    img_array = image_utils.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch

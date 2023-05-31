import tensorflow as tf

from img_dataset import *
import pathlib

from tensorflow.keras.preprocessing import image as image_utils

path = pathlib.Path("sample_img/")
num_classes = 2
img_width = 256
img_height = 256
batch_size = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=path,
    image_size=(img_height, img_width),
).map(preprocess_images)

val_split = 0.2
num_samples = dataset.cardinality().numpy()
num_val_samples = int(num_samples * val_split)
num_train_samples = num_samples - num_val_samples

train_dataset = dataset.take(num_train_samples)
val_dataset = dataset.skip(num_train_samples)


model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(train_dataset, epochs=20, validation_data=val_dataset)

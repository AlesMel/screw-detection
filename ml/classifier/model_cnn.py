import tensorflow as tf

from datetime import datetime
import pathlib

import seaborn as sns
import numpy as np

from tensorflow.keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib

logdir = "logs/training/" + datetime.now().strftime("%Y%m%d-%H%M%S")

path = pathlib.Path("sample_img_2/")
num_classes = 2
img_width = 512
img_height = 512
BATCH_SIZE = 16
IMG_SIZE = (img_width, img_height)

data_gen = ImageDataGenerator(
    validation_split=0.2,  # 20% of data will be used for validation
)

train_data = data_gen.flow_from_directory(
    "sample_img",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    color_mode="grayscale",
    shuffle=True,
    class_mode="categorical",
)

val_data = data_gen.flow_from_directory(
    "sample_img",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    color_mode="grayscale",
    shuffle=True,
    class_mode="categorical",
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)


callbacks = [
    ModelCheckpoint("../classifier_mdl.h5", save_best_only=True, monitor="val_loss"),
    TensorBoard(log_dir=logdir),
]

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_data, epochs=30, validation_data=val_data, callbacks=callbacks
)

# model = tf.keras.models.load_model("classifier_mdl.h5")
true_labels = val_data.classes
probabilities = model.predict(val_data)
predicted_labels = np.argmax(probabilities, axis=1)
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(2, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Validation confusion matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.show()


true_labels = train_data.classes
probabilities = model.predict(train_data)
predicted_labels = np.argmax(probabilities, axis=1)
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(2, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Train confusion matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.show()

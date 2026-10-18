import pandas as pd

data = pd.read_csv("train.csv")

print(data.head())
print(data.shape)
print(data.columns)

print("Total images:", len(data))
print("Missing values:")
print(data.isnull().sum())

print("Disease classes:")
print(data["label"].value_counts())

import matplotlib.pyplot as plt

data["label"].value_counts().plot(kind="bar")

plt.title("CropSense - Rice Disease Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

import os

for folder in os.listdir("train_images"):
    path = os.path.join("train_images", folder)

    if os.path.isdir(path):
        images = os.listdir(path)
        print(folder, ":", len(images), "images")

from PIL import Image
import os

folder = "train_images/normal"
image_name = os.listdir(folder)[0]
image_path = os.path.join(folder, image_name)

img = Image.open(image_path)

print("Image name:", image_name)
print("Image size:", img.size)
print("Image mode:", img.mode)

img.show()

import matplotlib.pyplot as plt

plt.imshow(img)
plt.title("CropSense - Rice Leaf Image")
plt.axis("off")
plt.show()

import numpy as np

pixels = np.array(img)

print("Image shape:", pixels.shape)
print("First pixel RGB:", pixels[0, 0])
print("Minimum pixel value:", pixels.min())
print("Maximum pixel value:", pixels.max())

normalized_img = pixels / 255.0

print("Original pixel:", pixels[0, 0])
print("Normalized pixel:", normalized_img[0, 0])
print("Minimum:", normalized_img.min())
print("Maximum:", normalized_img.max())

import tensorflow as tf

dataset = tf.keras.utils.image_dataset_from_directory(
    "train_images",
    image_size=(224, 224),
    batch_size=16,
    label_mode="int",
    shuffle=True,
    seed=42
)

print("Number of classes:", len(dataset.class_names))
print("Classes:", dataset.class_names)

for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Labels shape:", labels.shape)



train_ds = tf.keras.utils.image_dataset_from_directory(
    "train_images",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=16
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "train_images",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=16
)

print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))
print("Classes:", train_ds.class_names)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=20,
    validation_steps=5
)
model.save("cropsense_test_model.keras")

print("Model saved successfully!")

loaded_model = tf.keras.models.load_model(
    "cropsense_test_model.keras"
)

class_names = train_ds.class_names

img_path = "train_images"

img = tf.keras.utils.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

predictions = loaded_model.predict(img_array)

predicted_index = np.argmax(predictions[0])
predicted_disease = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("Predicted disease:", predicted_disease)
print("Confidence:", round(float(confidence), 2), "%")

import os

for root, dirs, files in os.walk("train_images"):
    if "100002.jpg" in files:
        print("Image found:", os.path.join(root, "100002.jpg"))
        break

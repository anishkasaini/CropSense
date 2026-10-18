
import os
import json
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "cropsense_mobilenet.keras",
    safe_mode=False
)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

img_path = None

for root, dirs, files in os.walk("train_images"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            break
    if img_path:
        break

if img_path:
    img = tf.keras.utils.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    index = np.argmax(predictions[0])

    print("Image:", img_path)
    print("Predicted disease:", class_names[index])
    print("Confidence:", round(float(predictions[0][index] * 100), 2), "%")
else:
    print("No image found!")


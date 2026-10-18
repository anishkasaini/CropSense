
import os
import json
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = tf.keras.models.load_model(
    "cropsense_mobilenet.keras",
    custom_objects={
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    },
    compile=False
)

with open("class_names.json", "r") as file:
    class_names = json.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("leaf_image")

        if not file or not file.filename:
            error = "Please select a leaf image."

        else:
            try:
                image = Image.open(file.stream).convert("RGB")
                image.thumbnail((1600, 1600))

                image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    "uploaded_leaf.jpg"
                )
                image.save(image_path, format="JPEG")

                image = image.resize((224, 224))
                image_array = np.array(image, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]
                index = int(np.argmax(predictions))

                result = {
                    "disease": class_names[index].replace("_", " ").title(),
                    "confidence": round(float(predictions[index]) * 100, 2)
                }

                image_url = "/" + image_path.replace("\\", "/")

            except Exception:
                error = "Unable to process this image. Please try another."

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)

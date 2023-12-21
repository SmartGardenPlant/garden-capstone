import os
import tensorflow as tf
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from keras.layers import Dropout
import numpy as np
from PIL import Image

with tf.keras.utils.custom_object_scope({'FixedDropout': Dropout}):
    model = tf.keras.models.load_model("model.h5", compile=False)

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]


# model = load_model("model.h5", compile=False)
label = ["aloevera", "banana", "bilimbi", "cantaloupe", "cassava", "coconut", "corn", "cucumber", "curcuma", "eggplant", "galangal", "ginger", "guava", "kale", "longbeans", "mango", "melon", "orange", "paddy", "papaya", "peper chili", "pineapple", "pomelo", "shallot", "soybeans", "spinach", "sweet potatoes", "tobacco", "waterapple", "watermelon"]



@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            img = Image.open(image_path).convert("RGB")
            img = img.resize((160, 160))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 160, 160, 3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_names = label[index]
            # class_names = class_names[2:]
            # confidence_score = prediction[0][index]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting",
                },
                "data": {
                    "plant_types_prediction": class_names,
                    "confidence": float(index)
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Client side error"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405
    

if __name__ == "__main__":
    app.run()

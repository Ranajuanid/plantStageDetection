from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("plant_stage_model.h5")

labels = ['seedling', 'vegetative', 'flowering','germination','fruiting']
ideal_params = {
    'seedling': {'temp': 24, 'humidity': 80},
    'vegetative': {'temp': 28, 'humidity': 70},
    'flowering': {'temp': 30, 'humidity': 60},
    'germination': {'temp': 25, 'humidity': 74},
    'fruiting': {'temp': 34, 'humidity': 68}
}

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.expand_dims(img, axis=0)

@app.route("/upload", methods=['POST'])
def upload():
    img = preprocess(request.data)
    pred = model.predict(img)
    idx = np.argmax(pred)
    stage = labels[idx]
    return jsonify({'stage': stage, 'ideal': ideal_params[stage]})

if __name__ == "__main__":
    app.run()

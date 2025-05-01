# from flask import Flask, request, jsonify
# from PIL import Image
# import io
# import numpy as np
# import tensorflow as tf
# import gdown
# import os

# app = Flask(__name__)

# # Model download setup
# url = 'https://drive.google.com/uc?id=1JgyW-FKZ2rJtdwNez3y7ZhLMQtepDr3G'
# model_path = 'plant_stage_model.h5'

# # Check if the model exists, if not, download it
# if not os.path.exists(model_path):
#     print("📥 Downloading model from Google Drive...")
#     gdown.download(url, model_path, quiet=False)

# # Load the model
# model = tf.keras.models.load_model(model_path)

# # Labels and ideal parameters for plant stages
# labels = ['seedling', 'vegetative', 'flowering', 'germination', 'fruiting']
# ideal_params = {
#     'seedling': {'temp': 24, 'humidity': 80},
#     'vegetative': {'temp': 28, 'humidity': 70},
#     'flowering': {'temp': 30, 'humidity': 60},
#     'germination': {'temp': 25, 'humidity': 74},
#     'fruiting': {'temp': 34, 'humidity': 68}
# }

# # Preprocessing function for the input image
# def preprocess(img_bytes):
#     img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))
#     img = np.array(img) / 255.0
#     if img.shape[-1] == 4:
#         img = img[..., :3]
#     return np.expand_dims(img, axis=0)

# # Upload route to handle image prediction
# @app.route("/upload", methods=['POST'])
# def upload():
#     try:
#         img = preprocess(request.data)
#         pred = model.predict(img)
#         idx = np.argmax(pred)
#         stage = labels[idx]
#         return jsonify({'stage': stage, 'ideal': ideal_params[stage]})
#     except Exception as e:
#         print("❌ ERROR:", str(e))
#         return jsonify({'error': str(e)}), 500

# # Running the app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=10000)

from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import gdown
import os

app = Flask(__name__)

# Model download from Google Drive
url = 'https://drive.google.com/uc?id=1JgyW-FKZ2rJtdwNez3y7ZhLMQtepDr3G'
model_path = 'plant_stage_model.h5'

# Download model if not present
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Stage labels and their ideal environmental parameters
labels = ['seedling', 'vegetative', 'flowering', 'germination', 'fruiting']
ideal_params = {
    'seedling': {'temp': 24, 'humidity': 80},
    'vegetative': {'temp': 28, 'humidity': 70},
    'flowering': {'temp': 30, 'humidity': 60},
    'germination': {'temp': 25, 'humidity': 74},
    'fruiting': {'temp': 34, 'humidity': 68}
}

# Store the latest prediction result
latest_result = {
    'stage': 'unknown',
    'ideal': {'temp': 0, 'humidity': 0}
}

# Preprocess image
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.expand_dims(img, axis=0)

# Upload image for prediction (used by ESP32-CAM)
@app.route("/upload", methods=['POST'])
def upload():
    global latest_result
    try:
        img = preprocess(request.data)
        pred = model.predict(img)
        idx = np.argmax(pred)
        stage = labels[idx]
        latest_result = {'stage': stage, 'ideal': ideal_params[stage]}
        print(f"✅ Prediction: {stage} → {latest_result['ideal']}")
        return jsonify(latest_result)
    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

# Get the latest prediction (used by ESP32)
@app.route("/latest", methods=['GET'])
def get_latest():
    return jsonify(latest_result)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)


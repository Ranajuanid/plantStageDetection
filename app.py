# # from flask import Flask, request, jsonify
# # from PIL import Image
# # import io
# # import numpy as np
# # import tensorflow as tf
# # import gdown
# # import os

# # app = Flask(__name__)

# # # Model download setup
# # url = 'https://drive.google.com/uc?id=1JgyW-FKZ2rJtdwNez3y7ZhLMQtepDr3G'
# # model_path = 'plant_stage_model.h5'

# # # Check if the model exists, if not, download it
# # if not os.path.exists(model_path):
# #     print("📥 Downloading model from Google Drive...")
# #     gdown.download(url, model_path, quiet=False)

# # # Load the model
# # model = tf.keras.models.load_model(model_path)

# # # Labels and ideal parameters for plant stages
# # labels = ['seedling', 'vegetative', 'flowering', 'germination', 'fruiting']
# # ideal_params = {
# #     'seedling': {'temp': 24, 'humidity': 80},
# #     'vegetative': {'temp': 28, 'humidity': 70},
# #     'flowering': {'temp': 30, 'humidity': 60},
# #     'germination': {'temp': 25, 'humidity': 74},
# #     'fruiting': {'temp': 34, 'humidity': 68}
# # }

# # # Preprocessing function for the input image
# # def preprocess(img_bytes):
# #     img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))
# #     img = np.array(img) / 255.0
# #     if img.shape[-1] == 4:
# #         img = img[..., :3]
# #     return np.expand_dims(img, axis=0)

# # # Upload route to handle image prediction
# # @app.route("/upload", methods=['POST'])
# # def upload():
# #     try:
# #         img = preprocess(request.data)
# #         pred = model.predict(img)
# #         idx = np.argmax(pred)
# #         stage = labels[idx]
# #         return jsonify({'stage': stage, 'ideal': ideal_params[stage]})
# #     except Exception as e:
# #         print("❌ ERROR:", str(e))
# #         return jsonify({'error': str(e)}), 500

# # # Running the app
# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=10000)

from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import gdown
import os

app = Flask(__name__)

# Model URLs
url1 = 'https://drive.google.com/uc?id=1JgyW-FKZ2rJtdwNez3y7ZhLMQtepDr3G'  # plant stages detection
url2 = 'https://drive.google.com/uc?id=1dsQ_bQFj5eYPVSfr9rqAtvCBngl9a1Ow'  # plant disease detection

model_path_stages = 'plant_stage_model.h5'
model_path_diseases = 'plant_disease_model.h5'

# Download models if not present
if not os.path.exists(model_path_stages):
    gdown.download(url1, model_path_stages, quiet=False)

if not os.path.exists(model_path_diseases):
    gdown.download(url2, model_path_diseases, quiet=False)

# Load models
model_stages = tf.keras.models.load_model(model_path_stages)
model_diseases = tf.keras.models.load_model(model_path_diseases)

# Labels and mappings
labels_stages = ['seedling', 'vegetative', 'flowering', 'germination', 'fruiting']
ideal_params = {
    'seedling': {'temp': 24, 'humidity': 80},
    'vegetative': {'temp': 28, 'humidity': 70},
    'flowering': {'temp': 30, 'humidity': 60},
    'germination': {'temp': 25, 'humidity': 74},
    'fruiting': {'temp': 34, 'humidity': 68}
}

labels_diseases = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

disease_alert = {
    'Tomato___Bacterial_spot': {'Unhealthy': 1},
    'Tomato___Early_blight': {'Unhealthy': 1},
    'Tomato___Late_blight': {'Unhealthy': 1},
    'Tomato___Leaf_Mold': {'Unhealthy': 1},
    'Tomato___Septoria_leaf_spot': {'Unhealthy': 1},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'Unhealthy': 1},
    'Tomato___Target_Spot': {'Unhealthy': 1},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'Unhealthy': 1},
    'Tomato___Tomato_mosaic_virus': {'Unhealthy': 1},
    'Tomato___healthy': {'Healthy': 1}
}

# Store latest results
latest_result1 = {'stage': 'unknown', 'ideal': {'temp': 0, 'humidity': 0}}
latest_result2 = {'Disease': 'unknown', 'Health_state': {'Healthy': 0, 'Unhealthy': 0}}

# Preprocess image
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.expand_dims(img, axis=0)

# Upload image for prediction
@app.route("/upload", methods=['POST'])
def upload():
    global latest_result1, latest_result2
    try:
        img = preprocess(request.data)

        # Predict plant stage
        pred_stage = model_stages.predict(img)
        idx1 = np.argmax(pred_stage)
        stage = labels_stages[idx1]
        latest_result1 = {'stage': stage, 'ideal': ideal_params[stage]}
        print(f"✅ Stage: {stage} → {latest_result1['ideal']}")

        # Predict disease
        pred_disease = model_diseases.predict(img)
        idx2 = np.argmax(pred_disease)
        disease = labels_diseases[idx2]
        health_state = disease_alert.get(disease, {'Healthy': 0, 'Unhealthy': 0})
        latest_result2 = {'Disease': disease, 'Health_state': health_state}
        print(f"✅ Disease: {disease} → {health_state}")

        return jsonify({'stage': latest_result1, 'disease': latest_result2})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

# Endpoint to get the latest results
@app.route("/latest", methods=['GET'])
def get_latest():
    return jsonify({'stage': latest_result1, 'disease': latest_result2})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # default to 10000 if PORT not set
    app.run(host='0.0.0.0', port=port)



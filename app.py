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
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__)

# Load model from Google Drive if not exists
model_path = 'plant_stage_model.h5'
if not os.path.exists(model_path):
    gdown.download('https://drive.google.com/uc?id=1JgyW-FKZ2rJtdwNez3y7ZhLMQtepDr3G', model_path, quiet=False)
model = tf.keras.models.load_model(model_path)

# Stage labels and parameters
labels = ['seedling', 'vegetative', 'flowering', 'germination', 'fruiting']
ideal_params = {
    'seedling': {'temp': 24, 'humidity': 80, 'light': 80},
    'vegetative': {'temp': 28, 'humidity': 70, 'light': 85},
    'flowering': {'temp': 30, 'humidity': 60, 'light': 90},
    'germination': {'temp': 25, 'humidity': 74, 'light': 78},
    'fruiting': {'temp': 34, 'humidity': 68, 'light': 75}
}

# Setup Google Sheets logging
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
sheet = client.open('Plant Ideal Parameters').sheet1

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.expand_dims(img, axis=0)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        image_bytes = request.data
        img = preprocess(image_bytes)
        prediction = model.predict(img)
        stage = labels[np.argmax(prediction)]
        params = ideal_params[stage]

        # Log to Google Sheets
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([timestamp, stage, params['temp'], params['humidity'], params['light']])

        return jsonify({
            'stage': stage,
            'ideal': params,
            'timestamp': timestamp
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)


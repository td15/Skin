from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import numpy as np
import os
from PIL import Image
import io
import pickle

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.environ.get("MODEL_DIR", "/Users/isa/Downloads/deployment_files")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model_v2_8class.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.pkl")
TEMPERATURE_PATH = os.path.join(MODEL_DIR, "temperature.pkl")

# Load model from deployment_files bundle.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "rb") as class_file:
    class_names = pickle.load(class_file)

temperature = 1.0
if os.path.exists(TEMPERATURE_PATH):
    with open(TEMPERATURE_PATH, "rb") as temp_file:
        temp_data = pickle.load(temp_file)
        temperature = float(getattr(temp_data, "temperature", temp_data))

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_skin():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    img_file = request.files['image']

    try:
        img = Image.open(img_file.stream).convert("RGB")
        img = img.resize((300, 300))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = efficientnet_preprocess(img_array)

        logits = model(img_array, training=False)
        prediction = logits.numpy()[0] if hasattr(logits, "numpy") else model.predict(img_array, verbose=0)[0]
        if temperature != 1.0:
            prediction = tf.nn.softmax(prediction / temperature).numpy()
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
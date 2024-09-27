from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pylance



app = Flask(__name__, static_folder='index')
CORS(app)

# Define the path to your model
MODEL_PATH = 'C:/Users/iampr/OneDrive/Desktop/speechrecognition/speech_emotion_model.h5'


# Define emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the pre-trained model
model = None
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def extract_feature(audio):
    # Implement your feature extraction here
    # This should match the feature extraction used in training
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    try:
        audio, _ = librosa.load(file, sr=22050, duration=3)  # Load 3 seconds of audio
        feature = extract_feature(audio)
        feature = np.expand_dims(feature, axis=(0, -1))  # Reshape for model input
        prediction = model.predict(feature)
        emotion_index = np.argmax(prediction)
        emotion = emotions[emotion_index]
        
        return jsonify({'emotion': emotion}), 200
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/api/start_stream', methods=['GET'])
def start_stream():
    # This is a placeholder for real-time streaming
    # In a real application, you'd set up WebSocket here
    return jsonify({'message': 'Streaming started successfully'}), 200

if __name__ == '__main__':
    app.run(use_reloader=True, port=5000, threaded=True)
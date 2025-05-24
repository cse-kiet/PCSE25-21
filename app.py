from flask import Flask, request, render_template, jsonify
from datetime import datetime
import random
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = None

def load_model():
    global model
    try:
        # Adjust this path to where your model is saved
        model = tf.keras.models.load_model('fer2013_emotion_model.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

# Class names (emotion labels) - update these to match your model's classes
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to grayscale if it's not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Resize to 48x48 (model input size)
    resized = cv2.resize(gray, (48, 48))
    
    # Normalize pixel values to [0,1]
    normalized = resized / 255.0
    
    # Reshape to match model's expected input
    input_data = normalized.reshape(1, 48, 48, 1)
    
    return input_data

def predict_emotion(image):
    """Predict emotion from image"""
    if model is None:
        return {"error": "Model not loaded"}
        
    try:
        # Preprocess the image
        input_data = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(input_data)
        
        # Get the index of the highest probability
        predicted_class_index = np.argmax(predictions[0])
        
        # Get the emotion label and probability
        emotion = class_names[predicted_class_index]
        probability = float(predictions[0][predicted_class_index])
        
        # Get all probabilities for visualization
        all_probabilities = {}
        for i, class_name in enumerate(class_names):
            all_probabilities[class_name] = float(predictions[0][i])
        
        return {
            "emotion": emotion,
            "probability": probability,
            "probabilities": all_probabilities
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})
        
    file = request.files['image']
    
    # Read the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Get prediction
    result = predict_emotion(img)
    
    return jsonify(result)

@app.route('/analyze_webcam', methods=['POST'])
def analyze_webcam():
    try:
        # Get base64 encoded image from request
        image_data = request.json.get('image', '')
        if not image_data:
            return jsonify({"error": "No image data received"})
            
        # Remove header from base64 string if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
            
        # Decode base64 to image
        img_bytes = base64.b64decode(image_data)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # Get prediction
        result = predict_emotion(img)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})
# ===== DATABASE =====
psychologists = {
    "Delhi": [
        {"name": "Dr. Ananya Sharma", "specialty": "Anxiety & Depression", "phone": "011-2654 1234", "clinic": "MindCare Clinic, CP", "website": "mindcare.example.com"},
        {"name": "Dr. Rajiv Mehta", "specialty": "Stress Management", "phone": "011-2789 5678", "clinic": "HealMind Center, Saket", "website": "healmind.example.com"}
    ],
    "Noida": [
        {"name": "Dr. Kirti Jain", "specialty": "Workplace Burnout", "phone": "0120-456 7890", "clinic": "Tranquil Souls, Sector 62", "website": "tranquilsouls.example.com"}
    ],
    "Mumbai": [
        {"name": "Dr. Arjun Patel", "specialty": "Cognitive Therapy", "phone": "022-3344 5566", "clinic": "Peaceful Mind, Bandra", "website": "peacefulmind.example.com"}
    ],
    "Bangalore": [
        {"name": "Dr. Priya Reddy", "specialty": "Mindfulness", "phone": "080-4455 6677", "clinic": "Serene Space, Koramangala", "website": "serenespace.example.com"}
    ]
}

questions = [
    "Have you had trouble sleeping recently?",
    "Do you feel overwhelmed by your responsibilities?",
    "Are you easily irritated or angered?",
    "Have you lost interest in activities you used to enjoy?",
    "Do you feel tired all the time?",
    "Do you have difficulty concentrating?",
    "Do you feel anxious most of the day?",
    "Have your eating habits changed significantly?",
    "Do you feel hopeless about the future?",
    "Do you experience physical symptoms like headaches or stomachaches?",
    "Do you avoid social interactions?",
    "Do you feel restless or unable to sit still?",
    "Do you have negative thoughts about yourself?",
    "Do you feel like crying often?",
    "Do you feel detached from reality?",
    "Have you had thoughts of self-harm?",
    "Do you feel like you've lost control of your life?",
    "Do you use alcohol or drugs to cope?",
    "Do you feel lonely even when with people?",
    "Do you feel like you're failing at everything?"
]

mood_diary_entries = []

@app.route('/')
def home():
    return render_template("base.html")

@app.route('/stress-prediction')
def stress_page():
    return render_template("stress.html", questions=questions)

@app.route('/emotion-detection')
def emotion_page():
    return render_template("emotion.html")

@app.route('/mood-diary', methods=['GET', 'POST'])
def mood_diary():
    return render_template("mood.html", mood_diary_entries=mood_diary_entries)

@app.route('/tips')
def tips():
    return render_template("tips.html")

@app.route('/emergency')
def emergency():
    return render_template("emergency.html")

@app.route('/predict', methods=['POST'])
def predict_stress():
    data = request.get_json()
    score = sum(data['answers'])
    prediction = 1 if score >= 8 else 0
    location = data['location'].strip().title()
    therapists = psychologists.get(location, [])
    return jsonify({
        "prediction": prediction,
        "location": location,
        "therapists": therapists
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True)

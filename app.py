import os
import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import functions from existing scripts
from quick_test import load_image, predict_with_tflite, predict_with_keras
from tf_utils import suppress_tf_warnings

# Suppress TF warnings
suppress_tf_warnings()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class names and descriptions
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratosis & Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

RISK_LEVELS = {
    "akiec": {"level": "High", "color": "#d9534f", "description": "Requires immediate medical attention"},
    "bcc": {"level": "Moderate", "color": "#f0ad4e", "description": "Consult a dermatologist soon"},
    "bkl": {"level": "Low", "color": "#5bc0de", "description": "Monitor for changes"},
    "df": {"level": "Low", "color": "#5bc0de", "description": "Usually benign"},
    "mel": {"level": "High", "color": "#d9534f", "description": "Requires immediate medical attention"},
    "nv": {"level": "Low", "color": "#5bc0de", "description": "Usually benign"},
    "vasc": {"level": "Low", "color": "#5bc0de", "description": "Monitor for changes"}
}

def load_model():
    """Load the best available model"""
    model_paths = [
        "optimized_model.keras",
        "optimized_model.h5",
        "dermatological_diagnosis_model.h5",
        "best_model.h5",
        "simple_model.keras",
        "simple_model.tflite",
        "simple_cnn_model.tflite"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Using model: {model_path}")
            return model_path
    
    print("No models found, trying to create one...")
    # Try to create a simple model if none exists
    try:
        from simple_model_creator import create_and_save_simple_model
        create_and_save_simple_model()
        if os.path.exists("simple_cnn_model.tflite"):
            return "simple_cnn_model.tflite"
    except Exception as e:
        print(f"Error creating model: {e}")
    
    return None

def process_image(image_data, model_path):
    """Process image and get prediction results"""
    # Save the uploaded image
    filename = secure_filename(f"upload_{int(time.time())}.png")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Convert data URI to image file
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]
    
    image_binary = base64.b64decode(image_data)
    with open(filepath, "wb") as f:
        f.write(image_binary)
    
    # Make prediction based on model type
    try:
        if model_path.endswith('.tflite'):
            _, confidence, predictions = predict_with_tflite(model_path, filepath)
        else:
            _, confidence, predictions = predict_with_keras(model_path, filepath)
        
        # Get the predicted class
        pred_class_idx = np.argmax(predictions)
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(confidence)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [
            {
                "class": CLASS_NAMES[idx],
                "name": CLASS_DESCRIPTIONS[CLASS_NAMES[idx]],
                "probability": float(predictions[idx] * 100),
                "risk": RISK_LEVELS[CLASS_NAMES[idx]]
            }
            for idx in top_indices
        ]
        
        # Prepare result
        result = {
            "success": True,
            "filename": filename,
            "filepath": filepath.replace("\\", "/"),  # For web URLs
            "predicted_class": pred_class,
            "class_name": CLASS_DESCRIPTIONS[pred_class],
            "confidence": confidence,
            "risk_level": RISK_LEVELS[pred_class],
            "top_predictions": top_predictions
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "filename": filename,
            "filepath": filepath.replace("\\", "/")
        }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/api/models')
def get_models():
    """Get available models"""
    models = [f for f in os.listdir('.') if f.endswith(('.h5', '.keras', '.tflite'))]
    return jsonify({"models": models})

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    # Check if image was uploaded
    if 'image' not in request.json:
        return jsonify({"success": False, "error": "No image provided"})
    
    image_data = request.json['image']
    selected_model = request.json.get('model', None)
    
    # Use specified model or find the best one
    if selected_model and os.path.exists(selected_model):
        model_path = selected_model
    else:
        model_path = load_model()
    
    if not model_path:
        return jsonify({"success": False, "error": "No model available"})
    
    # Process the image and get results
    result = process_image(image_data, model_path)
    return jsonify(result)

@app.route('/api/classes')
def get_classes():
    """Return information about the classes"""
    class_info = []
    for class_name in CLASS_NAMES:
        class_info.append({
            "code": class_name,
            "name": CLASS_DESCRIPTIONS[class_name],
            "risk": RISK_LEVELS[class_name]
        })
    return jsonify({"classes": class_info})

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"success": False, "error": "File too large"}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Not found"}), 404

if __name__ == '__main__':
    import time
    # Import needed for process_image
    
    print("Starting Dermatological Diagnosis Web Interface")
    # Preload the model
    model_path = load_model()
    if model_path:
        print(f"Model loaded: {model_path}")
        app.config['MODEL_PATH'] = model_path
    else:
        print("Warning: No model found. The application may not work correctly.")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)

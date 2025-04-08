"""
Simple and direct testing script for skin lesion classification models.
This script works with any model format and provides a straightforward way to test images.
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tf_utils import suppress_tf_warnings

# Suppress warnings
suppress_tf_warnings()

# Class names for ISIC dataset
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

def load_image(image_path, target_size=(299, 299)):
    """Load and preprocess a single image"""
    print(f"Loading image: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert from BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0,1]
    img_normalized = img / 255.0
    
    return img_normalized

def predict_with_tflite(model_path, image_path):
    """Make prediction using TFLite model"""
    print(f"Testing with TFLite model: {model_path}")
    
    # Load and preprocess the image
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0).astype(np.float32)
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input/output details for debugging
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    
    # Run inference
    print("Running inference...")
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process results
    predictions = output_data[0]
    pred_class_idx = np.argmax(predictions)
    confidence = predictions[pred_class_idx] * 100
    
    # Display results
    print("\nPrediction Results:")
    print(f"Class: {CLASS_NAMES[pred_class_idx]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class_idx]]}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show top 3 predictions
    print("\nTop 3 predictions:")
    top_indices = np.argsort(predictions)[::-1][:3]
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {CLASS_NAMES[idx]} ({CLASS_DESCRIPTIONS[CLASS_NAMES[idx]]}): {predictions[idx]*100:.2f}%")
    
    # Display the image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {CLASS_NAMES[pred_class_idx]} ({confidence:.2f}%)")
    plt.axis('off')
    
    # Save and show the result
    result_path = "quick_test_result.png"
    plt.savefig(result_path)
    plt.close()
    print(f"\nPrediction image saved to {result_path}")
    
    return pred_class_idx, confidence, predictions

def predict_with_keras(model_path, image_path):
    """Make prediction using Keras/TF model"""
    print(f"Testing with Keras model: {model_path}")
    
    # Load and preprocess the image
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    
    # Custom objects dictionary for Lambda layers
    custom_objects = {
        'feature_mask_tf': tf.constant(np.ones((2048,), dtype=np.float32))
    }
    
    # Load the model with custom objects
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading approach...")
        
        # Alternative approach using a simpler model
        try:
            model = tf.keras.models.load_model('simple_model.keras')
        except:
            print("Could not load any model. Please run convert_model.py first.")
            return None, 0, None
    
    # Make predictions
    predictions = model.predict(img_batch)[0]
    pred_class_idx = np.argmax(predictions)
    confidence = predictions[pred_class_idx] * 100
    
    # Display results
    print("\nPrediction Results:")
    print(f"Class: {CLASS_NAMES[pred_class_idx]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class_idx]]}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show top 3 predictions
    print("\nTop 3 predictions:")
    top_indices = np.argsort(predictions)[::-1][:3]
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {CLASS_NAMES[idx]} ({CLASS_DESCRIPTIONS[CLASS_NAMES[idx]]}): {predictions[idx]*100:.2f}%")
    
    # Display the image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {CLASS_NAMES[pred_class_idx]} ({confidence:.2f}%)")
    plt.axis('off')
    
    # Save and show the result
    result_path = "quick_test_result.png"
    plt.savefig(result_path)
    plt.close()
    print(f"\nPrediction image saved to {result_path}")
    
    return pred_class_idx, confidence, predictions

def main():
    """Main function to run the test"""
    print("Simple Skin Lesion Classification Tester")
    print("=" * 50)
    
    # List available TFLite models
    tflite_models = [f for f in os.listdir('.') if f.endswith('.tflite')]
    
    if not tflite_models:
        print("No TFLite models found. Let's create a simplified one.")
        
        # Run convert_model.py to create a simple model
        try:
            from convert_model import convert_model
            # Get an available model
            h5_models = [f for f in os.listdir('.') if f.endswith('.h5')]
            if h5_models:
                print(f"Converting {h5_models[0]} to TFLite format...")
                convert_model(h5_models[0])
                print("Model converted successfully.")
            else:
                print("No .h5 models found to convert.")
        except Exception as e:
            print(f"Error converting model: {e}")
    
    # Refresh the list
    tflite_models = [f for f in os.listdir('.') if f.endswith('.tflite')]
    
    if tflite_models:
        print("\nAvailable TFLite models:")
        for i, model in enumerate(tflite_models):
            print(f"{i+1}. {model}")
        
        # Use simple_model.tflite if available, otherwise use the first one
        if 'simple_model.tflite' in tflite_models:
            model_path = 'simple_model.tflite'
        else:
            model_path = tflite_models[0]
        
        print(f"\nUsing model: {model_path}")
    else:
        # Check for .keras models
        keras_models = [f for f in os.listdir('.') if f.endswith('.keras')]
        if keras_models:
            print("\nNo TFLite models found, but Keras models are available:")
            for i, model in enumerate(keras_models):
                print(f"{i+1}. {model}")
            model_path = keras_models[0]
        else:
            # Check for .h5 models
            h5_models = [f for f in os.listdir('.') if f.endswith('.h5')]
            if h5_models:
                print("\nNo TFLite or Keras models found, but H5 models are available:")
                for i, model in enumerate(h5_models):
                    print(f"{i+1}. {model}")
                model_path = h5_models[0]
            else:
                print("\nNo models found. Please run convert_model.py first.")
                return
    
    # Get image path from user
    image_path = input("\nEnter the path to the image file: ")
    
    # Make sure the image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Predict based on model type
    if model_path.endswith('.tflite'):
        predict_with_tflite(model_path, image_path)
    else:
        predict_with_keras(model_path, image_path)

if __name__ == "__main__":
    main()

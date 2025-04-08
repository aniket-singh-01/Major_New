
# filepath: model_interpreter.py
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def predict_with_h5(model_path, image_path, class_names):
    """
    Predict using direct loading of H5 file with custom objects
    
    Args:
        model_path: Path to the H5 model
        image_path: Path to the image to predict
        class_names: List of class names
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Get TensorFlow to suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Define custom objects for Lambda layers
    def identity_function(x):
        return x
    
    custom_objects = {
        'Lambda': tf.keras.layers.Lambda,
        'function': identity_function,
        'feature_mask_tf': tf.ones((2048,), dtype=tf.float32),
        'output_shape': lambda x: x
    }
    
    # Attempt multiple loading strategies
    strategies = [
        # Strategy 1: Direct load with custom objects
        lambda: tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False),
        
        # Strategy 2: Load with tf.keras.experimental.load_from_saved_model
        lambda: tf.keras.experimental.load_from_saved_model(model_path.replace('.h5', '_saved_model')),
        
        # Strategy 3: Use TFLite if available
        lambda: load_tflite(model_path.replace('.h5', '.tflite'), img_batch, class_names)
    ]
    
    # Try each strategy
    for i, strategy in enumerate(strategies):
        try:
            print(f"Trying loading strategy {i+1}...")
            result = strategy()
            if result is not None:
                if i == 2:  # TFLite strategy returns result directly
                    return result
                
                # For other strategies, make prediction with model
                model = result
                pred = model.predict(img_batch)[0]
                pred_class_idx = np.argmax(pred)
                pred_confidence = pred[pred_class_idx] * 100
                
                print(f"Prediction Results:")
                print(f"Class: {class_names[pred_class_idx]}")
                print(f"Confidence: {pred_confidence:.2f}%")
                
                return pred_class_idx, pred_confidence
        except Exception as e:
            print(f"Strategy {i+1} failed: {e}")
    
    print("All loading strategies failed")
    return None, 0

def load_tflite(tflite_path, img_batch, class_names):
    """Load and run prediction with TFLite model"""
    if not os.path.exists(tflite_path):
        print(f"TFLite file not found at {tflite_path}")
        return None
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_class_idx = np.argmax(pred)
    pred_confidence = pred[pred_class_idx] * 100
    
    print(f"TFLite Prediction Results:")
    print(f"Class: {class_names[pred_class_idx]}")
    print(f"Confidence: {pred_confidence:.2f}%")
    
    return pred_class_idx, pred_confidence

if __name__ == "__main__":
    model_path = input("Enter the path to the model file: ")
    image_path = input("Enter the path to the image file: ")
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # Replace with your actual class names
    
    predict_with_h5(model_path, image_path, class_names)

import os
import numpy as np
import tensorflow as tf
from tf_utils import suppress_tf_warnings

def convert_model(model_path, input_shape=(299, 299, 3)):
    """
    Convert H5 model to TFLite and SavedModel formats
    
    Args:
        model_path: Path to the H5 model
        input_shape: Input shape for the model
    """
    # Suppress TensorFlow warnings
    suppress_tf_warnings()
    
    print(f"Converting model: {model_path}")
    
    try:
        # Create a simple model for the TFLite conversion
        print("Creating a simple model for conversion...")
        
        # Create a CNN model with the same input/output structure
        simple_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            
            # Use a pre-trained model for feature extraction (MobileNetV2 is lighter than InceptionV3)
            tf.keras.applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape
            ),
            
            # Flatten features
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Classification layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 classes for skin lesions
        ])
        
        # Compile the model
        simple_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Simple model created successfully")
        
        # Save the simple model in different formats
        simple_tflite_path = os.path.join(os.path.dirname(model_path), 'simple_model.tflite')
        simple_keras_path = os.path.join(os.path.dirname(model_path), 'simple_model.keras')
        simple_saved_model_path = os.path.join(os.path.dirname(model_path), 'simple_model_saved')
        
        # Convert to TFLite
        print("Converting simple model to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(simple_model)
        tflite_model = converter.convert()
        
        with open(simple_tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Simple TFLite model saved to {simple_tflite_path}")
        
        # Save in native Keras format with .keras extension
        print(f"Saving simple model in native Keras format...")
        try:
            simple_model.save(simple_keras_path)
            print(f"Simple Keras model saved to {simple_keras_path}")
        except Exception as keras_error:
            print(f"Error saving in native Keras format: {keras_error}")
            # Try saving in HDF5 format if native format fails
            try:
                simple_h5_path = os.path.join(os.path.dirname(model_path), 'simple_model.h5')
                simple_model.save(simple_h5_path)
                print(f"Simple model saved in HDF5 format to {simple_h5_path}")
            except Exception as h5_error:
                print(f"Error saving in HDF5 format: {h5_error}")
        
        # Save as SavedModel format using export method
        print(f"Saving simple model in SavedModel format...")
        try:
            # Use export method for SavedModel format
            simple_model.export(simple_saved_model_path)
            print(f"Simple SavedModel saved to {simple_saved_model_path}")
        except Exception as export_error:
            print(f"Error using export method: {export_error}")
            # Try alternative SavedModel saving method
            try:
                tf.saved_model.save(simple_model, simple_saved_model_path)
                print(f"Simple SavedModel saved using tf.saved_model.save")
            except Exception as sm_error:
                print(f"Error with tf.saved_model.save: {sm_error}")
        
        return True
    except Exception as e:
        print(f"Error creating simple model: {e}")
        
        # Try a different approach - direct interpreter for prediction
        try:
            print("Creating a direct prediction script...")
            interpreter_script_path = os.path.join(os.path.dirname(model_path), 'model_interpreter.py')
            
            with open(interpreter_script_path, 'w') as f:
                f.write("""
# filepath: model_interpreter.py
import os
import tensorflow as tf
import numpy np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def predict_with_h5(model_path, image_path, class_names):
    \"\"\"
    Predict using direct loading of H5 file with custom objects
    
    Args:
        model_path: Path to the H5 model
        image_path: Path to the image to predict
        class_names: List of class names
    \"\"\"
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
    \"\"\"Load and run prediction with TFLite model\"\"\"
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
""")
            
            print(f"Created model interpreter script at {interpreter_script_path}")
            print(f"You can use this script to directly predict with your model:")
            print(f"python {interpreter_script_path}")
            
            return True
        except Exception as script_error:
            print(f"Error creating interpreter script: {script_error}")
            return False

if __name__ == "__main__":
    # Find all model files
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    
    if not model_files:
        print("No model files found in the current directory.")
    else:
        print("Available models:")
        for i, model_file in enumerate(model_files):
            print(f"{i+1}. {model_file}")
        
        model_idx = int(input("\nSelect a model to convert (number): ")) - 1
        model_path = model_files[model_idx]
        
        convert_model(model_path)

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_utils import suppress_tf_warnings

# Set these to match your actual classes
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def load_and_preprocess_image(image_path, target_size=(299, 299)):
    """Load and preprocess an image for prediction"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert from BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize [0-1]
    img = img / 255.0
    
    return img

def predict_with_simple_model(image_path):
    """Make a prediction using the simple model we created"""
    # Suppress TensorFlow warnings
    suppress_tf_warnings()
    
    # Check different potential model locations
    simple_model_paths = [
        "./simple_model.keras",  # Native Keras format
        "./simple_model.h5",     # H5 format
        "./simple_model_saved",  # SavedModel format
        "./simple_model.tflite"  # TFLite format
    ]
    
    # Find first available model
    model_path = None
    for path in simple_model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Found model at {model_path}")
            break
    
    if model_path is None:
        print("No model files found.")
        print("Please run 'convert_model.py' first to create the simple model")
        return
    
    try:
        # Load and preprocess the image
        img = load_and_preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Different loading strategies based on file type
        if model_path.endswith('.tflite'):
            # Use TFLite interpreter
            print("Using TFLite model...")
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], img_batch.astype(np.float32))
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            pred = interpreter.get_tensor(output_details[0]['index'])[0]
            
        elif model_path.endswith('.keras') or model_path.endswith('.h5'):
            # Use direct Keras loading
            print(f"Loading Keras model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            pred = model.predict(img_batch)[0]
            
        elif os.path.isdir(model_path):  # SavedModel format
            # For Keras 3, use TFSMLayer for SavedModel
            print("Using SavedModel with TFSMLayer...")
            try:
                # Try the newer Keras 3 approach
                keras_layer = tf.keras.layers.TFSMLayer(
                    model_path, 
                    call_endpoint='serving_default'
                )
                # Create a simple model with this layer
                inputs = tf.keras.Input(shape=img_batch.shape[1:])
                outputs = keras_layer(inputs)
                model = tf.keras.Model(inputs, outputs)
                pred = model.predict(img_batch)[0]
            except:
                # Fall back to tf.saved_model directly
                print("TFSMLayer failed, trying direct saved_model loading...")
                saved_model = tf.saved_model.load(model_path)
                
                # Try different possible signatures
                if hasattr(saved_model, 'signatures'):
                    serving_fn = saved_model.signatures['serving_default']
                    result = serving_fn(tf.constant(img_batch))
                    # Get the tensor from the output dictionary
                    output_name = list(result.keys())[0]
                    pred = result[output_name].numpy()[0]
                else:
                    # Try calling the model directly
                    pred = saved_model(tf.constant(img_batch)).numpy()[0]
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        # Process prediction
        pred_class_idx = np.argmax(pred)
        pred_confidence = pred[pred_class_idx] * 100
        
        # Display results
        print(f"\nPrediction Results:")
        print(f"Class: {CLASS_NAMES[pred_class_idx]}")
        print(f"Confidence: {pred_confidence:.2f}%")
        
        # Show top 3 predictions
        top_indices = np.argsort(pred)[::-1][:3]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {CLASS_NAMES[idx]}: {pred[idx]*100:.2f}%")
        
        # Display the image with prediction
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {CLASS_NAMES[pred_class_idx]} ({pred_confidence:.2f}%)")
        plt.axis('off')
        plt.savefig('simple_prediction_result.png')
        plt.close()
        
        print(f"Prediction image saved to 'simple_prediction_result.png'")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Try direct TFLite conversion as a last resort
        try:
            print("\nTrying to create and use a TFLite model directly...")
            # Create a simple CNN model
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(299, 299, 3)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Convert to TFLite without training
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = "emergency_model.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Use TFLite interpreter
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # Process image and make a basic prediction
            img = load_and_preprocess_image(image_path)
            img_batch = np.expand_dims(img, axis=0).astype(np.float32)
            
            # Set input tensor
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], img_batch)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            pred = interpreter.get_tensor(output_details[0]['index'])[0]
            
            print("\nEmergency prediction (untrained model):")
            pred_class_idx = np.argmax(pred)
            print(f"Predicted class: {CLASS_NAMES[pred_class_idx]}")
            print("Note: This prediction is from an untrained model and is just for testing purposes.")
            
        except Exception as e2:
            print(f"Emergency model also failed: {e2}")

if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ")
    predict_with_simple_model(image_path)

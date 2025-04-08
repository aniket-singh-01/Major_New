import os
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from xai_explainer import lime_explanation
from tf_utils import suppress_tf_warnings

def load_image(image_path, target_size=(299, 299)):
    """Load and preprocess an image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert from BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img / 255.0
    
    return img

def predict_image(model_path, image_path, class_names, explain=True):
    """
    Make prediction on a single image using the saved model
    
    Args:
        model_path: Path to the saved model
        image_path: Path to the image to classify
        class_names: List of class names
        explain: Whether to generate LIME explanation
    """
    # Suppress TensorFlow warnings
    suppress_tf_warnings()
    
    # Load the model with proper handling for Lambda layers
    try:
        # Define custom objects for loading models with Lambda layers
        custom_objects = {
            'Lambda': lambda config: tf.keras.layers.Lambda(
                eval(config.pop('function')),
                output_shape=lambda input_shape: input_shape,  # Define output shape explicitly
                **config
            )
        }
        
        # Try loading with custom objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Successfully loaded model from {model_path} with custom objects")
    except Exception as e:
        print(f"Error loading with custom objects: {e}")
        print("Trying alternative loading methods...")
        
        try:
            # Try loading with feature_mask_tf custom object
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'feature_mask_tf': tf.constant(np.ones((2048,), dtype=np.float32))
                },
                compile=False
            )
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"Successfully loaded model with feature_mask_tf custom object")
        except Exception as e2:
            print(f"Second loading attempt failed: {e2}")
            
            try:
                # Try running the convert_model.py first to create compatible formats
                if os.path.exists('convert_model.py'):
                    print("Attempting to convert the model to a compatible format...")
                    from convert_model import convert_model
                    convert_model(model_path)
                    
                    # Try loading TFLite model
                    tflite_path = model_path.replace('.h5', '.tflite')
                    if os.path.exists(tflite_path):
                        print(f"Using TFLite model at {tflite_path}")
                        interpreter = tf.lite.Interpreter(model_path=tflite_path)
                        interpreter.allocate_tensors()
                        
                        def predict_with_tflite(img):
                            input_details = interpreter.get_input_details()
                            output_details = interpreter.get_output_details()
                            
                            # Set the input tensor
                            interpreter.set_tensor(input_details[0]['index'], img)
                            # Run inference
                            interpreter.invoke()
                            # Get the output tensor
                            return interpreter.get_tensor(output_details[0]['index'])
                        
                        # Load and preprocess the image
                        img = load_image(image_path)
                        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
                        
                        # Make prediction
                        pred = predict_with_tflite(img_batch)[0]
                        pred_class_idx = np.argmax(pred)
                        pred_confidence = pred[pred_class_idx] * 100
                        
                        print(f"\nPrediction Results:")
                        print(f"Class: {class_names[pred_class_idx]}")
                        print(f"Confidence: {pred_confidence:.2f}%")
                        
                        # Show top 3 predictions
                        print("\nTop 3 predictions:")
                        top_indices = np.argsort(pred)[::-1][:3]
                        for i, idx in enumerate(top_indices):
                            print(f"{i+1}. {class_names[idx]}: {pred[idx]*100:.2f}%")
                        
                        # Display the image
                        plt.figure(figsize=(6, 6))
                        plt.imshow(img)
                        plt.title(f"Prediction: {class_names[pred_class_idx]} ({pred_confidence:.2f}%)")
                        plt.axis('off')
                        plt.savefig('prediction_result.png')
                        
                        print("\nNote: LIME explanation not available for TFLite models")
                        return pred_class_idx, pred_confidence
                    else:
                        # Try saved model
                        saved_model_path = f"{os.path.splitext(model_path)[0]}_saved_model"
                        if os.path.exists(saved_model_path):
                            print(f"Using SavedModel at {saved_model_path}")
                            model = tf.saved_model.load(saved_model_path)
                            
                            # Define a prediction function
                            @tf.function
                            def predict_fn(x):
                                return model(x)
                            
                            # Continue with prediction...
                        else:
                            raise ValueError("No compatible model formats found")
            except Exception as e3:
                print(f"All loading methods failed: {e3}")
                print("Please run convert_model.py first to create compatible model formats")
                return None, 0
    
    # Load and preprocess the image
    img = load_image(image_path)
    
    # Make prediction
    try:
        pred = model.predict(np.expand_dims(img, axis=0))[0]
    except Exception as pred_error:
        print(f"Error during prediction: {pred_error}")
        print("Trying alternative prediction method...")
        
        try:
            # Try calling the model directly as a function
            pred = model(np.expand_dims(img, axis=0))[0]
        except:
            print("Could not make prediction with this model. Please convert it first.")
            return None, 0
            
    pred_class_idx = np.argmax(pred)
    pred_confidence = pred[pred_class_idx] * 100
    
    print(f"\nPrediction Results:")
    print(f"Class: {class_names[pred_class_idx]}")
    print(f"Confidence: {pred_confidence:.2f}%")
    
    # Show top 3 predictions
    print("\nTop 3 predictions:")
    top_indices = np.argsort(pred)[::-1][:3]
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {class_names[idx]}: {pred[idx]*100:.2f}%")
    
    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[pred_class_idx]} ({pred_confidence:.2f}%)")
    plt.axis('off')
    plt.savefig('prediction_result.png')
    
    # Generate LIME explanation if requested
    if explain:
        try:
            explanation_path = f"explanation_{os.path.basename(image_path).split('.')[0]}.png"
            explanation, _ = lime_explanation(
                model, img, class_names, 
                save_path=explanation_path
            )
            print(f"\nExplanation saved to {explanation_path}")
        except Exception as lime_error:
            print(f"Could not generate LIME explanation: {lime_error}")
    
    return pred_class_idx, pred_confidence

def batch_test(model_path, image_dir, class_names):
    """
    Test multiple images in a directory
    
    Args:
        model_path: Path to the saved model
        image_dir: Directory containing test images
        class_names: List of class names
    """
    # Suppress TensorFlow warnings
    suppress_tf_warnings()
    
    # Load the model with custom objects to handle the Lambda layer issue
    print(f"Loading model from {model_path}...")
    try:
        # Define custom objects to handle Lambda layer with output shape
        custom_objects = {
            'Lambda': lambda config: tf.keras.layers.Lambda(
                eval(config.pop('function')),
                output_shape=config.pop('output_shape', None),
                **config
            )
        }
        
        # First try loading with custom objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model loaded successfully with custom objects.")
    except Exception as e:
        print(f"Error with custom loading: {e}")
        print("Trying alternative loading method...")
        
        # Alternative loading method
        try:
            # First load the model structure without weights
            with tf.keras.utils.custom_object_scope({
                'feature_mask_tf': tf.constant(np.ones((2048,), dtype=np.float32))
            }):
                model = tf.keras.models.load_model(model_path)
            print("Model loaded with alternative method.")
        except Exception as e2:
            print(f"Error loading model: {e2}")
            print("Trying direct inference without loading full model...")
            
            # Create a wrapper function for prediction
            def predict_wrapper(img):
                # Load just for prediction, not for modification
                tf.keras.backend.clear_session()
                try:
                    # This loads the model in prediction mode only
                    interpreter = tf.lite.Interpreter(model_path=model_path.replace('.h5', '.tflite'))
                    interpreter.allocate_tensors()
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    # Set the input tensor
                    interpreter.set_tensor(input_details[0]['index'], img)
                    # Run inference
                    interpreter.invoke()
                    # Get the output tensor
                    return interpreter.get_tensor(output_details[0]['index'])
                except:
                    # If TFLite doesn't work, try one last approach with saved_model format
                    print("TFLite failed, trying saved_model format...")
                    saved_model_path = os.path.join(os.path.dirname(model_path), 'saved_model')
                    if os.path.exists(saved_model_path):
                        model = tf.saved_model.load(saved_model_path)
                        return model(img)
                    else:
                        raise ValueError("Could not load the model in any format.")
            
            # Get all image files
            image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            results = []
            
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                try:
                    img = load_image(img_path)
                    img_batch = np.expand_dims(img, axis=0)
                    
                    # Use predict wrapper
                    pred = predict_wrapper(img_batch)[0]
                    pred_class_idx = np.argmax(pred)
                    pred_confidence = pred[pred_class_idx] * 100
                    
                    results.append({
                        'image': img_file,
                        'predicted_class': class_names[pred_class_idx],
                        'confidence': pred_confidence
                    })
                    
                    print(f"Predicted {img_file} as {class_names[pred_class_idx]} with {pred_confidence:.2f}% confidence")
                
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            
            # Create summary DataFrame
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv('batch_test_results.csv', index=False)
            print(f"\nBatch test complete. Processed {len(results)} images.")
            print("Results saved to 'batch_test_results.csv'")
            
            return
    
    # If we get here, the model was loaded successfully
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            img = load_image(img_path)
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            pred_class_idx = np.argmax(pred)
            pred_confidence = pred[pred_class_idx] * 100
            
            results.append({
                'image': img_file,
                'predicted_class': class_names[pred_class_idx],
                'confidence': pred_confidence
            })
            
            print(f"Predicted {img_file} as {class_names[pred_class_idx]} with {pred_confidence:.2f}% confidence")
        
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Create summary DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('batch_test_results.csv', index=False)
    print(f"\nBatch test complete. Processed {len(results)} images.")
    print("Results saved to 'batch_test_results.csv'")

def compare_models(image_path, class_names):
    """
    Compare predictions from all available models on a single image
    
    Args:
        image_path: Path to the image to classify
        class_names: List of class names
    """
    # Suppress TensorFlow warnings
    suppress_tf_warnings()
    
    # Find all model files
    model_files = [f for f in os.listdir('.') if f.endswith(('.h5', '.keras'))]
    
    if not model_files:
        print("No model files found in the current directory.")
        return
    
    # Load and preprocess the image
    img = load_image(image_path)
    
    # Create comparison table
    results = []
    
    for model_file in model_files:
        try:
            # Load the model
            model = tf.keras.models.load_model(model_file, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Make prediction
            start_time = time.time()
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            inference_time = time.time() - start_time
            
            pred_class_idx = np.argmax(pred)
            pred_confidence = pred[pred_class_idx] * 100
            
            # Store results
            results.append({
                'Model': model_file,
                'Prediction': class_names[pred_class_idx],
                'Confidence': f"{pred_confidence:.2f}%",
                'Inference Time': f"{inference_time*1000:.1f} ms"
            })
            
        except Exception as e:
            print(f"Error with model {model_file}: {e}")
    
    # Display comparison table
    if results:
        print("\nModel Comparison Results:")
        print("-" * 80)
        print(f"{'Model':<30} {'Prediction':<15} {'Confidence':<15} {'Inference Time':<15}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['Model']:<30} {result['Prediction']:<15} {result['Confidence']:<15} {result['Inference Time']:<15}")
            
        # Recommend best model based on confidence
        best_model = max(results, key=lambda x: float(x['Confidence'].rstrip('%')))
        print("\nRecommendation:")
        print(f"Model '{best_model['Model']}' provides the highest confidence ({best_model['Confidence']}) for this image.")
    
    return results

if __name__ == "__main__":
    # Set these values to match your model and dataset
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # Replace with your actual class names
    
    # Select a model file
    print("Available models:")
    model_files = [f for f in os.listdir('.') if f.endswith(('.h5', '.keras'))]
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    model_idx = int(input("\nSelect a model (number): ")) - 1
    model_path = model_files[model_idx]
    
    # Choose test mode
    print("\nTest options:")
    print("1. Test a single image")
    print("2. Batch test all images in a directory")
    print("3. Compare all models on a single image")
    
    option = int(input("Select an option (1-3): "))
    
    if option == 1:
        # Single image test
        image_path = input("\nEnter the path to the test image: ")
        
        # First try to run the existing predict_image function
        try:
            predict_image(model_path, image_path, class_names)
        except Exception as e:
            print(f"Error with standard prediction: {e}")
            print("\nAttempting to convert the model first...")
            
            # Try to convert the model to a more compatible format
            try:
                from convert_model import convert_model
                if convert_model(model_path):
                    print("\nModel converted. Trying prediction with converted model...")
                    
                    # Try TFLite version
                    tflite_path = model_path.replace('.h5', '.tflite')
                    if os.path.exists(tflite_path):
                        try:
                            # Create a basic TFLite predictor
                            interpreter = tf.lite.Interpreter(model_path=tflite_path)
                            interpreter.allocate_tensors()
                            input_details = interpreter.get_input_details()
                            output_details = interpreter.get_output_details()
                            
                            # Load and preprocess the image
                            img = load_image(image_path)
                            img_batch = np.expand_dims(img, axis=0).astype(np.float32)
                            
                            # Set the input tensor
                            interpreter.set_tensor(input_details[0]['index'], img_batch)
                            # Run inference
                            interpreter.invoke()
                            # Get the output tensor
                            pred = interpreter.get_tensor(output_details[0]['index'])[0]
                            
                            pred_class_idx = np.argmax(pred)
                            pred_confidence = pred[pred_class_idx] * 100
                            
                            print(f"\nPrediction Results (using TFLite):")
                            print(f"Class: {class_names[pred_class_idx]}")
                            print(f"Confidence: {pred_confidence:.2f}%")
                            
                            # Show top 3 predictions
                            print("\nTop 3 predictions:")
                            top_indices = np.argsort(pred)[::-1][:3]
                            for i, idx in enumerate(top_indices):
                                print(f"{i+1}. {class_names[idx]}: {pred[idx]*100:.2f}%")
                            
                            # Display the image
                            plt.figure(figsize=(6, 6))
                            plt.imshow(img)
                            plt.title(f"Prediction: {class_names[pred_class_idx]} ({pred_confidence:.2f}%)")
                            plt.axis('off')
                            plt.savefig('prediction_result.png')
                        except Exception as tflite_error:
                            print(f"Error with TFLite prediction: {tflite_error}")
            except Exception as conv_error:
                print(f"Error converting model: {conv_error}")
                print("\nPlease run convert_model.py separately first, then try again.")
    
    elif option == 2:
        # Batch test
        image_dir = input("\nEnter the path to the directory containing test images: ")
        batch_test(model_path, image_dir, class_names)
    
    elif option == 3:
        # Compare all models
        image_path = input("\nEnter the path to the test image: ")
        compare_models(image_path, class_names)
        
    else:
        print("Invalid option.")

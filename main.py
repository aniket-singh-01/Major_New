import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from data_processor import load_isic_data, preprocess_data, create_synthetic_dataset
from model_builder import build_model, train_model
from gwo_optimizer import GreyWolfOptimizer
from xai_explainer import lime_explanation
import matplotlib.pyplot as plt
import sys

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("Starting Dermatological Diagnosis System")
    
    # Step 1: Load and preprocess the ISIC 2019 dataset
    print("Loading and preprocessing data...")
    
    # First, try to find the dataset directory
    data_dir = "./dataset"
    
    # If dataset directory doesn't exist, prompt the user
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found at {data_dir}")
        print("Please enter the absolute path to your dataset directory:")
        user_path = input().strip()
        
        if os.path.exists(user_path):
            data_dir = user_path
        else:
            print(f"Path {user_path} does not exist.")
            print("Would you like to continue with a synthetic dataset for testing? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Exiting program.")
                sys.exit(0)
            else:
                X_train, X_val, X_test, y_train, y_val, y_test, class_names = create_synthetic_dataset(
                    num_classes=8, img_size=(299, 299))
                use_synthetic = True
    
    try:
        # Try to load the dataset
        use_synthetic = False
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_isic_data(data_dir)
        
        # Apply data preprocessing (augmentation, etc.)
        print("Applying data preprocessing and augmentation...")
        try:
            datagen, X_train, y_train, X_val, y_val = preprocess_data(X_train, y_train, X_val, y_val)
        except Exception as preproc_error:
            print(f"Warning: Data preprocessing error: {preproc_error}")
            print("Continuing without data augmentation.")
            datagen = None
        
        # Display dataset information
        print(f"Dataset loaded successfully:")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {class_names}")
        
        # Step 2: Build the Inception V3 based model
        print("Building Inception V3 model...")
        base_model = InceptionV3(weights='imagenet', include_top=False, 
                               input_shape=(299, 299, 3))
        model = build_model(base_model, len(class_names))
        
        # Step 3: Train the model
        print("Training the model...")
        batch_size = min(32, X_train.shape[0] // 10)  # Adjust batch size based on dataset size
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=batch_size)
        
        # Step 4: Evaluate the model
        print("Evaluating the model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Step 5: Apply Grey Wolf Optimizer for feature selection
        print("Applying Grey Wolf Optimizer for feature selection...")
        gwo = GreyWolfOptimizer(model, X_train, y_train, X_val, y_val)
        optimized_model = gwo.optimize()
        
        # Evaluate optimized model
        opt_test_loss, opt_test_accuracy = optimized_model.evaluate(X_test, y_test)
        print(f"Optimized model test accuracy: {opt_test_accuracy:.4f}")
        
        # Step 6: Apply LIME for explainability
        print("Generating LIME explanations...")
        sample_idx = np.random.randint(0, len(X_test))
        sample_image = X_test[sample_idx]
        true_label = np.argmax(y_test[sample_idx])
        
        explanation, pred_label = lime_explanation(optimized_model, sample_image, class_names)
        
        print(f"True label: {class_names[true_label]}")
        print(f"Predicted label: {class_names[pred_label]}")
        
        # Save the optimized model
        optimized_model.save('dermatological_diagnosis_model.h5')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Would you like to continue with a synthetic dataset for testing? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            X_train, X_val, X_test, y_train, y_val, y_test, class_names = create_synthetic_dataset(
                num_classes=8, img_size=(299, 299))
            
            # Continue with the model building and training
            print("Building model with synthetic data...")
            base_model = InceptionV3(weights='imagenet', include_top=False, 
                                   input_shape=(299, 299, 3))
            model = build_model(base_model, len(class_names))
            
            # Use smaller epochs for synthetic data
            history = train_model(model, X_train, y_train, X_val, y_val, epochs=5)
            
            # Continue with evaluation and other steps...
            test_loss, test_accuracy = model.evaluate(X_test, y_test)
            print(f"Test accuracy with synthetic data: {test_accuracy:.4f}")
            
            # Save the model
            model.save('synthetic_model.h5')
            print("Synthetic model saved successfully!")
        else:
            print("Exiting program.")
            sys.exit(1)

if __name__ == "__main__":
    main()

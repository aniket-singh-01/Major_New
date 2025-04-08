import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from model_results_analyzer import analyze_model_predictions
from data_processor import load_isic_data, preprocess_data, create_synthetic_dataset
from model_builder import build_model, train_model
from gwo_optimizer import GreyWolfOptimizer
from xai_explainer import lime_explanation
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from debugging import check_data_balance, evaluate_model, plot_training_history
# Import the warning suppression function
from tf_utils import suppress_tf_warnings

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # Suppress TensorFlow warnings
    suppress_tf_warnings()
    
    # Ensure TensorFlow is running eagerly
    tf.config.run_functions_eagerly(True)
    print("Eager execution enabled:", tf.executing_eagerly())
    
    print("Starting Dermatological Diagnosis System")
    
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
    
    # Ask the user which model to run
    print("\nPlease select a model type to run:")
    print("1. Inception V3 transfer learning model (for skin lesion images)")
    print("2. Simple neural network model (for any input)")
    print("3. Continue from last checkpoint")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    # Initialize variables to avoid undefined errors
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = None, None, None, None, None, None, None
    
    try:
        # Load the dataset regardless of model choice
        print("Loading and preprocessing data...")
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
        
        # Check data balance
        check_data_balance(y_train)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Would you like to continue with a synthetic dataset for testing? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            X_train, X_val, X_test, y_train, y_val, y_test, class_names = create_synthetic_dataset(
                num_classes=8, img_size=(299, 299))
            use_synthetic = True
        else:
            print("Exiting program.")
            sys.exit(1)
    
    # Process the user's choice
    if choice == '1':
        # Run the Inception V3 model
        run_inception_model(X_train, X_val, X_test, y_train, y_val, y_test, class_names)
    elif choice == '2':
        # Run the simple model
        run_simple_model(X_train, X_val, X_test, y_train, y_val, y_test, class_names)
    elif choice == '3':
        # Load and continue from last checkpoint
        continue_from_checkpoint(X_train, X_val, X_test, y_train, y_val, y_test, class_names)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

def run_inception_model(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    """Run the Inception V3 transfer learning model"""
    # Step 2: Build the Inception V3 based model
    print("\nBuilding Inception V3 model...")
    base_model = InceptionV3(weights='imagenet', include_top=False, 
                           input_shape=(299, 299, 3))
    model = build_model(base_model, len(class_names))
    
    # Step 3: Train the model
    print("Training the model...")
    batch_size = min(32, X_train.shape[0] // 10)  # Adjust batch size based on dataset size
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=batch_size)
    
    # Step 4: Evaluate the model
    print("Evaluating the model...")
    evaluation_results = model.evaluate(X_test, y_test)
    test_accuracy = evaluation_results[1] if isinstance(evaluation_results, list) else evaluation_results
    print(f"Test accuracy: {test_accuracy:.4f}")
    analyze_model_predictions(model, X_test, y_test, class_names)
    
    # Visualize results
    plot_training_history(history)
    evaluate_model(model, X_test, y_test, class_names)
    
    # Optional: Apply Grey Wolf Optimizer and LIME
    try_advanced_features(model, X_train, X_val, X_test, y_train, y_val, y_test, class_names)

def run_simple_model(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    """Run the simpler neural network model from model_fix.py"""
    # Print relevant information for debugging
    input_shape = X_train.shape[1:]  # Get the shape excluding the batch dimension
    num_classes = y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train))
    
    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Import from model_fix to avoid confusion with model_builder
    from model_fix import create_model, train_model
    
    print("Creating and training simple neural network model...")
    model = create_model(input_shape, num_classes)
    
    # Train model with fewer epochs for testing
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=10)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluation_results = model.evaluate(X_test, y_test)
    test_accuracy = evaluation_results[1] if isinstance(evaluation_results, list) else evaluation_results
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Visualize results
    plot_training_history(history)
    evaluate_model(model, X_test, y_test, class_names)
    
    # Save the model
    model.save('simple_model.h5')
    print("Model saved successfully!")

def continue_from_checkpoint(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    """Continue training from the last saved checkpoint"""
    checkpoint_path = 'best_model.h5'
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Would you like to start a new model? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            print("Please choose a model type:")
            print("1. Inception V3 model")
            print("2. Simple neural network")
            model_choice = input("Enter choice (1/2): ").strip()
            if model_choice == '1':
                run_inception_model(X_train, X_val, X_test, y_train, y_val, y_test, class_names)
            elif model_choice == '2':
                run_simple_model(X_train, X_val, X_test, y_train, y_val, y_test, class_names)
            else:
                print("Invalid choice. Exiting.")
                sys.exit(1)
        else:
            print("Exiting program.")
            sys.exit(0)
        return
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Try loading with custom objects first
        custom_objects = {
            'feature_mask_tf': tf.constant(np.ones((2048,), dtype=np.float32))
        }
        model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading approach...")
        
        # Try regular loading
        try:
            model = tf.keras.models.load_model(checkpoint_path)
        except Exception as e2:
            print(f"Could not load model: {e2}")
            print("Please choose a different option.")
            return
    
    # Ensure data is in the right format for TensorFlow
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    
    # Continue training with a more direct approach
    print("Continuing training...")
    
    # Use a simpler training approach that avoids the error
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'continued_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate and visualize
    print("Evaluating the model...")
    evaluation_results = model.evaluate(X_test, y_test)
    test_accuracy = evaluation_results[1] if isinstance(evaluation_results, list) else evaluation_results
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Import required functions directly
    from debugging import plot_training_history, evaluate_model
    
    # Visualize results
    plot_training_history(history)
    evaluate_model(model, X_test, y_test, class_names)
    
    # Save the updated model
    model.save('continued_model.h5')
    print("Updated model saved successfully!")

def try_advanced_features(model, X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    """Optionally apply advanced features like GWO and LIME"""
    print("\nWould you like to try advanced features? (y/n)")
    response = input().strip().lower()
    
    if response != 'y':
        return
    
    try:
        # Step 5: Apply Grey Wolf Optimizer for feature selection
        print("Applying Grey Wolf Optimizer for feature selection...")
        gwo = GreyWolfOptimizer(model, X_train, y_train, X_val, y_val)
        optimized_model = gwo.optimize()
        
        # Evaluate optimized model
        evaluation_results = optimized_model.evaluate(X_test, y_test)
        opt_test_accuracy = evaluation_results[1] if isinstance(evaluation_results, list) else evaluation_results
        print(f"Optimized model test accuracy: {opt_test_accuracy:.4f}")
        
        # Step 6: Apply LIME for explainability
        print("Generating LIME explanations...")
        sample_idx = np.random.randint(0, len(X_test))
        sample_image = X_test[sample_idx]
        true_label = np.argmax(y_test[sample_idx]) if len(y_test.shape) > 1 else y_test[sample_idx]
        
        # Save path for explanation
        lime_save_path = 'lime_explanation_result.png'
        explanation, pred_label = lime_explanation(
            optimized_model, sample_image, class_names, 
            save_path=lime_save_path
        )
        
        print(f"True label: {class_names[true_label]}")
        print(f"Predicted label: {class_names[pred_label]}")
        print(f"Explanation saved to {lime_save_path}")
        
        # Save the optimized model - use .keras format to suppress warning
        try:
            # Try the newer .keras format first
            model_path = 'optimized_model.keras'
            optimized_model.save(model_path)
            print(f"Optimized model saved successfully to {model_path}")
        except Exception as save_error:
            # Fall back to HDF5 format if .keras format fails
            print(f"Could not save in .keras format: {save_error}")
            model_path = 'optimized_model.h5'
            optimized_model.save(model_path)
            print(f"Optimized model saved in HDF5 format to {model_path}")
    
    except Exception as e:
        print(f"Error in advanced features: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with basic model...")

if __name__ == "__main__":
    main()

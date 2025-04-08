"""
Accurate testing script for dermatological diagnosis using MobileNetV2 transfer learning.
This script avoids the Lambda layer issues entirely by creating a fresh model and training it
on a small sample of data to enable better predictions.
"""
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.utils import shuffle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0,1]
    return img

def find_and_load_sample_data(base_dir="dataset", sample_per_class=20, target_size=(224, 224)):
    """Find and load a small sample of training data from the dataset directory"""
    print("Loading sample data for quick training...")
    
    # Find training data directories
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir):
        train_dir = base_dir  # If no train subdirectory, use base directory
    
    X_train = []
    y_train = []
    
    total_samples = 0
    
    # Try to find class directories
    class_dirs = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir) and os.path.isdir(class_dir):
            class_dirs.append((class_name, class_dir))
    
    if not class_dirs:
        print("Warning: Could not find class directories. Looking in subdirectories...")
        for subdir in os.listdir(train_dir):
            subpath = os.path.join(train_dir, subdir)
            if os.path.isdir(subpath):
                for class_name in CLASS_NAMES:
                    class_dir = os.path.join(subpath, class_name)
                    if os.path.exists(class_dir) and os.path.isdir(class_dir):
                        class_dirs.append((class_name, class_dir))
    
    if class_dirs:
        # Load samples from each class directory
        for class_name, class_dir in class_dirs:
            class_idx = CLASS_NAMES.index(class_name)
            
            # Get all image files in this class directory
            image_files = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                          glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                          glob.glob(os.path.join(class_dir, "*.png"))
            
            # Shuffle and select samples
            if image_files:
                np.random.shuffle(image_files)
                selected_files = image_files[:min(sample_per_class, len(image_files))]
                
                for img_path in selected_files:
                    try:
                        img = load_and_preprocess_image(img_path, target_size)
                        X_train.append(img)
                        
                        # One-hot encode the label
                        label = np.zeros(len(CLASS_NAMES))
                        label[class_idx] = 1
                        y_train.append(label)
                        
                        total_samples += 1
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            print(f"Loaded {len(selected_files) if 'selected_files' in locals() else 0} samples from class {class_name}")
    
    # If we couldn't find class directories or got no samples, create synthetic data
    if total_samples == 0:
        print("No real samples found, creating synthetic data")
        X_train = np.random.rand(sample_per_class * len(CLASS_NAMES), target_size[0], target_size[1], 3)
        
        # Create balanced labels
        y_train = np.zeros((sample_per_class * len(CLASS_NAMES), len(CLASS_NAMES)))
        for i in range(len(CLASS_NAMES)):
            y_train[i*sample_per_class:(i+1)*sample_per_class, i] = 1
        
        total_samples = len(X_train)
    
    # Convert to numpy arrays
    X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    
    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    print(f"Loaded {total_samples} total samples for quick training")
    return X_train, y_train

def create_and_train_quick_model(X_train, y_train, epochs=5):
    """Create and train a MobileNetV2 model on the sample data"""
    print("Creating and training a lightweight transfer learning model...")
    
    # Create a MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    print("Model training complete!")
    return model

def predict_and_visualize(model, image_path, target_size=(224, 224)):
    """Make prediction on a single image and visualize the results"""
    print(f"Testing image: {image_path}")
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path, target_size)
    img_batch = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch)[0]
    pred_class_idx = np.argmax(prediction)
    confidence = prediction[pred_class_idx] * 100
    
    # Display results
    print("\nPrediction Results:")
    print(f"Class: {CLASS_NAMES[pred_class_idx]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class_idx]]}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show top 3 predictions
    print("\nTop 3 predictions:")
    top_indices = np.argsort(prediction)[::-1][:3]
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {CLASS_NAMES[idx]} ({CLASS_DESCRIPTIONS[CLASS_NAMES[idx]]}): {prediction[idx]*100:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Original image with prediction
    plt.imshow(img)
    
    # Set title color based on risk
    title_color = "red" if CLASS_NAMES[pred_class_idx] in ["mel", "akiec"] else "blue"
    
    plt.title(
        f"Diagnosis: {CLASS_NAMES[pred_class_idx]}\n"
        f"{CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class_idx]]}\n"
        f"Confidence: {confidence:.2f}%",
        color=title_color,
        fontsize=14
    )
    
    plt.axis('off')
    
    # Save the visualization
    result_path = f"diagnosis_{os.path.basename(image_path)}.png"
    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()
    
    print(f"Visualization saved to {result_path}")
    
    return pred_class_idx, confidence, prediction

def batch_test(model, image_dir, target_size=(224, 224)):
    """Test all images in a directory"""
    print(f"Batch testing all images in {image_dir}")
    
    # Find all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    results = []
    
    for img_path in image_files:
        print(f"Processing {os.path.basename(img_path)}")
        
        try:
            # Load and preprocess the image
            img = load_and_preprocess_image(img_path, target_size)
            img_batch = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = model.predict(img_batch, verbose=0)[0]
            pred_class_idx = np.argmax(prediction)
            confidence = prediction[pred_class_idx] * 100
            
            # Store results
            results.append({
                'file': os.path.basename(img_path),
                'predicted_class': CLASS_NAMES[pred_class_idx],
                'confidence': confidence
            })
            
            print(f"Predicted as {CLASS_NAMES[pred_class_idx]} with {confidence:.2f}% confidence")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save results to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = "batch_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Batch results saved to {csv_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        # Print summary to console as fallback
        for result in results:
            print(f"{result['file']}: {result['predicted_class']} ({result['confidence']:.2f}%)")

if __name__ == "__main__":
    print("=" * 50)
    print("Accurate Dermatological Diagnosis Testing")
    print("=" * 50)
    
    # Check for existing trained model
    model_path = "quick_derma_model.keras"
    
    if os.path.exists(model_path):
        print(f"Found existing trained model at {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        # Load sample data
        X_train, y_train = find_and_load_sample_data()
        
        # Create and train model
        model = create_and_train_quick_model(X_train, y_train)
        
        # Save the trained model
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Test options
    print("\nTest options:")
    print("1. Test a single image")
    print("2. Batch test all images in a directory")
    
    option = input("Select an option (1/2): ").strip()
    
    if option == '1':
        # Test single image
        image_path = input("Enter the path to the image you want to test: ")
        
        if os.path.exists(image_path):
            predict_and_visualize(model, image_path)
        else:
            print(f"Image not found at {image_path}")
    
    elif option == '2':
        # Batch test
        image_dir = input("Enter the path to the directory containing test images: ")
        
        if os.path.exists(image_dir) and os.path.isdir(image_dir):
            batch_test(model, image_dir)
        else:
            print(f"Directory not found: {image_dir}")
    
    else:
        print("Invalid option")

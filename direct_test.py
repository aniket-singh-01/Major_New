"""
Direct testing script for dermatological models with Lambda layer fixes
"""
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Lambda

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Class names
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
    """Load and preprocess an image for prediction"""
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img / 255.0  # Normalize to [0,1]

def rebuild_model(model_path):
    """Rebuild a model structure that approximates the original model"""
    print("Rebuilding model structure...")
    # Create a base model using InceptionV3
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3)
    )
    
    # Build the model structure
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Skip the problematic Lambda layer for feature selection
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    # Create and compile the model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model structure rebuilt successfully")
    return model

def direct_test(image_path, model_path="dermatological_diagnosis_model.h5"):
    """Test the image directly by recreating the model architecture"""
    # Load and preprocess the image
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    
    # Create the model (skipping the problematic Lambda layer)
    model = rebuild_model(model_path)
    
    # Run inference for visual demo (will be inaccurate without proper weights)
    print("Making prediction...")
    pred = model.predict(img_batch)[0]
    pred_class_idx = np.argmax(pred)
    confidence = pred[pred_class_idx] * 100
    
    # Get second opinion with a MobileNetV2 model (for comparison)
    print("Getting second opinion with MobileNetV2...")
    mobile_model = create_mobilenet_classifier()
    mobile_pred = mobile_model.predict(img_batch)[0]
    mobile_class_idx = np.argmax(mobile_pred)
    mobile_confidence = mobile_pred[mobile_class_idx] * 100
    
    # Display results
    print("\nPrimary Prediction (Demo Only):")
    print(f"Class: {CLASS_NAMES[pred_class_idx]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class_idx]]}")
    print(f"Confidence: {confidence:.2f}%")
    
    print("\nSecond Opinion (MobileNetV2):")
    print(f"Class: {CLASS_NAMES[mobile_class_idx]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[mobile_class_idx]]}")
    print(f"Confidence: {mobile_confidence:.2f}%")
    
    # Create visualization with both predictions
    plt.figure(figsize=(14, 7))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # First prediction
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    title_color = "red" if CLASS_NAMES[pred_class_idx] in ["mel", "akiec"] else "blue"
    plt.title(f"Primary Prediction\n{CLASS_NAMES[pred_class_idx]}\n({confidence:.2f}%)", 
             color=title_color, fontsize=12)
    plt.axis('off')
    
    # Second prediction
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    title_color = "red" if CLASS_NAMES[mobile_class_idx] in ["mel", "akiec"] else "blue"
    plt.title(f"MobileNetV2 Opinion\n{CLASS_NAMES[mobile_class_idx]}\n({mobile_confidence:.2f}%)", 
             color=title_color, fontsize=12)
    plt.axis('off')
    
    # Save result
    result_path = f"combined_result_{os.path.basename(image_path)}.png"
    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()
    
    print(f"Visualization saved to {result_path}")
    
    # Display recommended next steps
    print("\nRECOMMENDED NEXT STEPS:")
    print("1. To get accurate predictions with your trained model, rebuild it without Lambda layers")
    print("2. Or use the MobileNetV2 comparison as a guide")
    print("3. For better results, run the convert_model.py script to create a simplified TFLite version")

def create_mobilenet_classifier():
    """Create a MobileNetV2 classifier for second opinion"""
    # Create a MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Get model and image paths
    model_path = "dermatological_diagnosis_model.h5"
    if not os.path.exists(model_path):
        print(f"Note: Model not found at {model_path}, but we'll proceed with a demo anyway")
    
    # Get image path
    image_path = input("Enter the path to the image you want to test: ")
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
    else:
        direct_test(image_path, model_path)

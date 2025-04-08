"""
Simple, focused script to test the dermatological_diagnosis_model.h5
"""
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

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

def test_image(model_path, image_path):
    """Test a single image with the dermatological model"""
    # Load and preprocess the image
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        # Try with custom objects for Lambda layer
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'feature_mask_tf': tf.constant(np.ones((2048,), dtype=np.float32))
            },
            compile=False
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure that the model file exists and is accessible")
        return
    
    # Make prediction
    print("Making prediction...")
    pred = model.predict(img_batch)[0]
    pred_class_idx = np.argmax(pred)
    confidence = pred[pred_class_idx] * 100
    
    # Display results
    print("\nPrediction Results:")
    print(f"Class: {CLASS_NAMES[pred_class_idx]} - {CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class_idx]]}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show top 3 predictions
    print("\nTop 3 predictions:")
    top_indices = np.argsort(pred)[::-1][:3]
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {CLASS_NAMES[idx]} ({CLASS_DESCRIPTIONS[CLASS_NAMES[idx]]}): {pred[idx]*100:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    title_color = "red" if CLASS_NAMES[pred_class_idx] in ["mel", "akiec"] else "blue"
    plt.title(f"Prediction: {CLASS_NAMES[pred_class_idx]}\n({CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class_idx]]})\nConfidence: {confidence:.2f}%", 
             color=title_color, fontsize=12)
    plt.axis('off')
    
    # Save result
    result_path = f"result_{os.path.basename(image_path)}.png"
    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()
    
    print(f"Visualization saved to {result_path}")

if __name__ == "__main__":
    # Default model path - you can change this to point directly to your model
    model_path = "dermatological_diagnosis_model.h5"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        model_path = input("Enter the path to your dermatological model: ")
    
    # Get image path
    image_path = input("Enter the path to the image you want to test: ")
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
    else:
        test_image(model_path, image_path)

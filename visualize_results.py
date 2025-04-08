"""
Visualization and analysis script for dermatology model performance
This script creates detailed visualizations of model performance and feature importance
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
import glob
from tf_utils import suppress_tf_warnings

# Suppress warnings
suppress_tf_warnings()

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

# Risk levels for color coding
RISK_LEVELS = {
    "akiec": "High",
    "bcc": "Moderate",
    "bkl": "Low",
    "df": "Low",
    "mel": "High",
    "nv": "Low",
    "vasc": "Low"
}

def load_and_preprocess_images(image_dir, target_size=(299, 299), limit=100):
    """Load and preprocess images from a directory with class structure"""
    # Check if directory exists
    if not os.path.exists(image_dir):
        raise ValueError(f"Directory not found: {image_dir}")
    
    images = []
    labels = []
    filenames = []
    
    # Process each class directory
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} not found")
            continue
        
        # Get image files
        image_files = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                    glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                    glob.glob(os.path.join(class_dir, "*.png"))
        
        # Limit the number of images per class
        if limit > 0:
            np.random.shuffle(image_files)
            image_files = image_files[:min(limit, len(image_files))]
        
        print(f"Loading {len(image_files)} images from class {class_name}")
        
        # Process each image
        for img_path in image_files:
            try:
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                img = img / 255.0  # Normalize
                
                images.append(img)
                
                # Create one-hot encoded label
                label = np.zeros(len(CLASS_NAMES))
                label[class_idx] = 1
                labels.append(label)
                
                # Store filename for reference
                filenames.append(os.path.basename(img_path))
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Loaded {len(images)} images total")
    return X, y, filenames

def load_best_model():
    """Load the best available model for testing"""
    # Look for models in this order of preference
    model_paths = [
        "optimized_model.keras",
        "optimized_model.h5",
        "dermatological_diagnosis_model.h5",
        "best_model.h5",
        "simple_model.keras",
        "simple_model.tflite"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Found model: {model_path}")
            
            try:
                if model_path.endswith(".tflite"):
                    # Load TFLite model
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    
                    # Create a function that mimics the predict method
                    def predict_fn(images):
                        results = []
                        for img in images:
                            input_details = interpreter.get_input_details()
                            output_details = interpreter.get_output_details()
                            interpreter.set_tensor(input_details[0]['index'], 
                                                 np.expand_dims(img, axis=0).astype(np.float32))
                            interpreter.invoke()
                            output = interpreter.get_tensor(output_details[0]['index'])
                            results.append(output[0])
                        return np.array(results)
                    
                    # Return the prediction function and model type
                    return predict_fn, "tflite", model_path
                else:
                    # Load Keras/H5 model
                    custom_objects = {
                        'feature_mask_tf': tf.constant(np.ones((2048,), dtype=np.float32))
                    }
                    
                    model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects=custom_objects, 
                        compile=False
                    )
                    
                    # Return the model's predict method and model type
                    return model.predict, "keras", model_path
            
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
                continue
    
    raise ValueError("No valid models found")

def visualize_predictions(model_fn, X, y, filenames, output_dir="visualization_results"):
    """Visualize model predictions on a test set"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model_fn(X)
    
    # Convert to class indices
    y_true = np.argmax(y, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred_classes)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Create classification report
    report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC for each class
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
    # Plot precision-recall curves
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y[:, i], y_pred[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
    plt.close()
    
    # Visualize a few examples (correct and incorrect predictions)
    visualize_example_predictions(X, y_true, y_pred, y_pred_classes, filenames, output_dir)
    
    # Create per-class accuracy table
    class_metrics = []
    for i, class_name in enumerate(CLASS_NAMES):
        class_indices = (y_true == i)
        class_accuracy = np.mean(y_pred_classes[class_indices] == i) if np.any(class_indices) else 0
        class_count = np.sum(class_indices)
        
        class_metrics.append({
            'Class': class_name,
            'Description': CLASS_DESCRIPTIONS[class_name],
            'Samples': class_count,
            'Accuracy': f"{class_accuracy:.4f}",
            'F1-Score': f"{report[class_name]['f1-score']:.4f}",
            'Precision': f"{report[class_name]['precision']:.4f}",
            'Recall': f"{report[class_name]['recall']:.4f}"
        })
    
    metrics_df = pd.DataFrame(class_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'class_metrics.csv'), index=False)
    
    print(f"Results saved to {output_dir} directory")
    
    return y_pred, y_true, y_pred_classes

def visualize_example_predictions(X, y_true, y_pred, y_pred_classes, filenames, output_dir):
    """Visualize example predictions (correct and incorrect)"""
    # Create subdirectories
    correct_dir = os.path.join(output_dir, 'correct_predictions')
    incorrect_dir = os.path.join(output_dir, 'incorrect_predictions')
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    # Get indices of correct and incorrect predictions
    correct_indices = np.where(y_true == y_pred_classes)[0]
    incorrect_indices = np.where(y_true != y_pred_classes)[0]
    
    # Randomly select examples to visualize
    n_examples = min(5, len(correct_indices), len(incorrect_indices))
    if n_examples == 0:
        print("No examples to visualize")
        return
    
    # Randomly select indices
    if len(correct_indices) > 0:
        selected_correct = np.random.choice(correct_indices, size=n_examples, replace=False)
    else:
        selected_correct = []
    
    if len(incorrect_indices) > 0:
        selected_incorrect = np.random.choice(incorrect_indices, size=n_examples, replace=False)
    else:
        selected_incorrect = []
    
    # Visualize correct predictions
    for i, idx in enumerate(selected_correct):
        img = X[idx]
        true_label = y_true[idx]
        pred_label = y_pred_classes[idx]
        confidence = y_pred[idx][pred_label] * 100
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"True: {CLASS_NAMES[true_label]}\nPredicted: {CLASS_NAMES[pred_label]} ({confidence:.2f}%)", 
                 color='green')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(correct_dir, f"correct_{i+1}_{filenames[idx]}"))
        plt.close()
    
    # Visualize incorrect predictions
    for i, idx in enumerate(selected_incorrect):
        img = X[idx]
        true_label = y_true[idx]
        pred_label = y_pred_classes[idx]
        confidence = y_pred[idx][pred_label] * 100
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"True: {CLASS_NAMES[true_label]}\nPredicted: {CLASS_NAMES[pred_label]} ({confidence:.2f}%)", 
                 color='red')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(incorrect_dir, f"incorrect_{i+1}_{filenames[idx]}"))
        plt.close()
    
    # Create summary of most common confusions
    if len(incorrect_indices) > 0:
        confusions = {}
        for idx in incorrect_indices:
            true_label = y_true[idx]
            pred_label = y_pred_classes[idx]
            # Use '->' instead of Unicode arrow to avoid encoding issues
            key = f"{CLASS_NAMES[true_label]} -> {CLASS_NAMES[pred_label]}"
            
            if key in confusions:
                confusions[key] += 1
            else:
                confusions[key] = 1
        
        # Sort by frequency
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        
        # Write to file with UTF-8 encoding to handle special characters
        with open(os.path.join(output_dir, 'common_confusions.txt'), 'w', encoding='utf-8') as f:
            f.write("Most common confusions:\n")
            for confusion, count in sorted_confusions[:10]:  # Top 10
                f.write(f"{confusion}: {count} instances\n")

def analyze_class_difficulty(y_true, y_pred, y_pred_classes, output_dir):
    """Analyze which classes are most difficult for the model"""
    # Calculate per-class metrics
    class_metrics = []
    
    for i, class_name in enumerate(CLASS_NAMES):
        # Get samples of this class
        class_indices = (y_true == i)
        if not np.any(class_indices):
            continue
            
        # Calculate metrics
        correct = (y_pred_classes[class_indices] == i)
        accuracy = np.mean(correct)
        
        # Calculate average confidence
        confidences = np.max(y_pred[class_indices], axis=1)
        avg_confidence = np.mean(confidences)
        
        # Calculate confidence of correct and incorrect predictions
        if np.any(correct):
            correct_conf = np.mean(confidences[correct])
        else:
            correct_conf = 0
            
        if np.any(~correct):
            incorrect_conf = np.mean(confidences[~correct])
        else:
            incorrect_conf = 0
        
        class_metrics.append({
            'Class': class_name,
            'Description': CLASS_DESCRIPTIONS[class_name],
            'Accuracy': accuracy,
            'Avg Confidence': avg_confidence,
            'Correct Confidence': correct_conf,
            'Incorrect Confidence': incorrect_conf,
            'Risk Level': RISK_LEVELS[class_name]
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(class_metrics)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'class_difficulty.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Bar chart of accuracy by class
    plt.bar(df['Class'], df['Accuracy'], alpha=0.7)
    plt.axhline(y=np.mean(df['Accuracy']), color='r', linestyle='--', label='Average')
    
    # Add risk level as color
    for i, (_, row) in enumerate(df.iterrows()):
        if row['Risk Level'] == 'High':
            plt.gca().get_children()[i].set_color('red')
        elif row['Risk Level'] == 'Moderate':
            plt.gca().get_children()[i].set_color('orange')
        else:
            plt.gca().get_children()[i].set_color('green')
    
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy.png'))
    plt.close()
    
    return df

def main():
    print("Dermatological Diagnosis Model Visualization and Analysis")
    print("=" * 60)
    
    # Load test data
    try:
        # First, try to load from 'dataset/valid' directory
        data_dir = "dataset/valid"
        if os.path.exists(data_dir):
            X, y, filenames = load_and_preprocess_images(data_dir, limit=30)  # 30 images per class max
        else:
            # Otherwise, prompt user for directory
            data_dir = input("Enter path to test data directory with class subdirectories: ")
            if not os.path.exists(data_dir):
                print(f"Directory not found: {data_dir}")
                return
            X, y, filenames = load_and_preprocess_images(data_dir, limit=30)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Load best model
    try:
        model_fn, model_type, model_path = load_best_model()
        print(f"Using model: {model_path} (type: {model_type})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory
    output_dir = "visualization_results"
    
    # Visualize predictions
    y_pred, y_true, y_pred_classes = visualize_predictions(model_fn, X, y, filenames, output_dir)
    
    # Analyze class difficulty
    difficulty_df = analyze_class_difficulty(y_true, y_pred, y_pred_classes, output_dir)
    
    # Print recommendations
    print("\nAnalysis Complete! Key findings:")
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred_classes)
    print(f"- Overall accuracy: {accuracy:.2%}")
    
    # Easiest and hardest classes
    if not difficulty_df.empty:
        easiest = difficulty_df.loc[difficulty_df['Accuracy'].idxmax()]
        hardest = difficulty_df.loc[difficulty_df['Accuracy'].idxmin()]
        
        print(f"- Easiest class: {easiest['Class']} ({easiest['Accuracy']:.2%} accuracy)")
        print(f"- Most challenging class: {hardest['Class']} ({hardest['Accuracy']:.2%} accuracy)")
        
        # High risk classes with low accuracy
        high_risk_low_acc = difficulty_df[(difficulty_df['Risk Level'] == 'High') & 
                                        (difficulty_df['Accuracy'] < 0.7)]
        
        if not high_risk_low_acc.empty:
            print("\nAttention needed for these high-risk classes with lower accuracy:")
            for _, row in high_risk_low_acc.iterrows():
                print(f"- {row['Class']} ({row['Description']}): {row['Accuracy']:.2%} accuracy")
    
    print(f"\nDetailed visualizations and metrics saved to {output_dir} directory")

if __name__ == "__main__":
    main()

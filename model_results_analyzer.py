import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def analyze_model_predictions(model, X_test, y_test, class_names):
    """
    Analyze model predictions with detailed metrics
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert to class indices
    y_test_indices = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    # Create confusion matrix
    create_confusion_matrix(y_test_indices, y_pred_indices, class_names)
    
    # Plot ROC curve
    plot_roc_curves(y_test, y_pred, class_names)
    
    # Plot precision-recall curve
    plot_precision_recall_curves(y_test, y_pred, class_names)
    
    # Analyze confidence distribution
    analyze_confidence_distribution(y_pred, y_test_indices, y_pred_indices, class_names)
    
    # Analyze misclassifications
    analyze_misclassifications(X_test, y_test_indices, y_pred_indices, y_pred, class_names)

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_detailed.png')
    plt.close()

def plot_roc_curves(y_test, y_pred, class_names):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 10))
    
    # Compute ROC curve and ROC area for each class
    for i, class_name in enumerate(class_names):
        if len(y_test.shape) > 1:  # One-hot encoded
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc = roc_auc_score(y_test[:, i], y_pred[:, i])
        else:  # Class indices
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), 
                                    y_pred[:, i] if len(y_pred.shape) > 1 else (y_pred == i).astype(int))
            roc_auc = roc_auc_score((y_test == i).astype(int), 
                                    y_pred[:, i] if len(y_pred.shape) > 1 else (y_pred == i).astype(int))
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

def plot_precision_recall_curves(y_test, y_pred, class_names):
    """Plot precision-recall curves for each class"""
    plt.figure(figsize=(12, 10))
    
    for i, class_name in enumerate(class_names):
        if len(y_test.shape) > 1:  # One-hot encoded
            precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
            ap = average_precision_score(y_test[:, i], y_pred[:, i])
        else:  # Class indices
            precision, recall, _ = precision_recall_curve((y_test == i).astype(int), 
                                                        y_pred[:, i] if len(y_pred.shape) > 1 else (y_pred == i).astype(int))
            ap = average_precision_score((y_test == i).astype(int), 
                                        y_pred[:, i] if len(y_pred.shape) > 1 else (y_pred == i).astype(int))
        
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.savefig('precision_recall_curves.png')
    plt.close()

def analyze_confidence_distribution(y_pred, y_true, y_pred_indices, class_names):
    """Analyze confidence distribution of predictions"""
    # Get confidence scores
    confidence = np.max(y_pred, axis=1)
    
    # Separate correct and incorrect predictions
    correct = y_true == y_pred_indices
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(confidence[correct], bins=20, label='Correct predictions', alpha=0.7)
    sns.histplot(confidence[~correct], bins=20, label='Incorrect predictions', alpha=0.7)
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.savefig('confidence_distribution.png')
    plt.close()
    
    # Detailed statistics per class
    class_data = []
    for i, class_name in enumerate(class_names):
        class_samples = y_true == i
        if np.sum(class_samples) > 0:
            avg_conf = np.mean(y_pred[class_samples, i])
            correct_in_class = (y_pred_indices[class_samples] == i)
            class_accuracy = np.mean(correct_in_class) if len(correct_in_class) > 0 else 0
            
            class_data.append({
                'Class': class_name,
                'Samples': np.sum(class_samples),
                'Avg Confidence': avg_conf,
                'Accuracy': class_accuracy
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(class_data)
    df.to_csv('class_statistics.csv', index=False)
    print("Class statistics saved to 'class_statistics.csv'")

def analyze_misclassifications(X_test, y_true, y_pred, y_pred_probs, class_names):
    """Analyze the most common misclassifications"""
    # Find misclassified samples
    misclassified = y_true != y_pred
    
    if np.sum(misclassified) == 0:
        print("No misclassifications found!")
        return
    
    # Create a confusion table of true vs predicted for misclassified samples
    confusion_pairs = {}
    for i in range(len(y_true[misclassified])):
        true_idx = y_true[misclassified][i]
        pred_idx = y_pred[misclassified][i]
        pair = (class_names[true_idx], class_names[pred_idx])
        
        if pair in confusion_pairs:
            confusion_pairs[pair] += 1
        else:
            confusion_pairs[pair] = 1
    
    # Sort by frequency
    confusion_pairs = {k: v for k, v in sorted(confusion_pairs.items(), 
                                              key=lambda item: item[1], reverse=True)}
    
    # Print most common misclassifications
    print("\nMost common misclassifications:")
    for (true_class, pred_class), count in list(confusion_pairs.items())[:5]:
        print(f"True: {true_class}, Predicted: {pred_class}, Count: {count}")
    
    # Save misclassification summary to CSV
    misclass_data = []
    for (true_class, pred_class), count in confusion_pairs.items():
        misclass_data.append({
            'True Class': true_class,
            'Predicted Class': pred_class,
            'Count': count
        })
    
    misclass_df = pd.DataFrame(misclass_data)
    misclass_df.to_csv('misclassification_summary.csv', index=False)
    print("Misclassification summary saved to 'misclassification_summary.csv'")

# Example usage:
# analyze_model_predictions(model, X_test, y_test, class_names)

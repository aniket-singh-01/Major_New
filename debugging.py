import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def check_data_balance(y_train):
    """Check if classes are imbalanced"""
    if len(y_train.shape) > 1:  # If one-hot encoded
        class_counts = np.sum(y_train, axis=0)
    else:
        class_counts = np.bincount(y_train)
    
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} samples ({count/len(y_train)*100:.2f}%)")
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_counts)), class_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class Distribution')
    plt.savefig('class_distribution.png')
    plt.close()
    
    return class_counts

def evaluate_model(model, X_test, y_test, class_names=None):
    """Evaluate model performance with detailed metrics"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names if class_names else range(cm.shape[1]),
               yticklabels=class_names if class_names else range(cm.shape[0]))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test_classes, y_pred_classes,
        target_names=class_names if class_names else [str(i) for i in range(len(np.unique(y_test_classes)))]
    ))
    
    return y_pred

def plot_training_history(history):
    """Plot training and validation metrics"""
    metrics = ['accuracy', 'loss', 'auc', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.subplot(2, 3, i+1)
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Training and Validation {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Example usage:
# check_data_balance(y_train)
# history = model.fit(...)
# plot_training_history(history)
# evaluate_model(model, X_test, y_test, class_names=['class1', 'class2', ...])

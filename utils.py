import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
import itertools

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history object from model.fit()
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, classes):
    """
    Plot ROC curve for multi-class classification
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_prob: Predicted probabilities
        classes: Class names
    """
    n_classes = len(classes)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_true_binary = np.argmax(y_true, axis=1)
    
    for i in range(n_classes):
        # True binary labels for this class
        y_binary = (y_true_binary == i).astype(int)
        # Predicted probabilities for this class
        y_score = y_pred_prob[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_binary, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    
    colors = itertools.cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown'])
    
    for i, color, cls in zip(range(n_classes), colors, classes):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {cls} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def generate_classification_report(y_true, y_pred, class_names):
    """
    Generate and print classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
    """
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Save to file
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    return report

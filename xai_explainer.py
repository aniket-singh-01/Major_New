import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing import image

def lime_explanation(model, img_array, class_names, num_samples=1000, top_features=5, save_path=None):
    """
    Generate LIME explanation for a model prediction
    
    Args:
        model: The trained model
        img_array: Image to explain (single image)
        class_names: List of class names
        num_samples: Number of samples for LIME
        top_features: Number of top features to highlight
        save_path: Path to save the explanation image
        
    Returns:
        explanation: The LIME explanation object
        pred_label: The predicted label
    """
    # Check if image is preprocessed (values between 0-1)
    if img_array.max() <= 1.0:
        img_array_for_display = (img_array * 255).astype('uint8')
    else:
        img_array_for_display = img_array.astype('uint8')
    
    # Create explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_array, 
        model.predict, 
        top_labels=len(class_names), 
        hide_color=0, 
        num_samples=num_samples
    )
    
    # Get model prediction
    pred = model.predict(np.expand_dims(img_array, axis=0))[0]
    pred_label = np.argmax(pred)
    pred_confidence = pred[pred_label] * 100
    
    print(f"Model predicted: {class_names[pred_label]} with {pred_confidence:.2f}% confidence")
    
    # Get the explanation for the top prediction
    temp, mask = explanation.get_image_and_mask(
        pred_label, 
        positive_only=True, 
        num_features=top_features, 
        hide_rest=False
    )
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    ax1.imshow(img_array_for_display)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Explanation
    img_boundary = mark_boundaries(temp/255.0, mask)
    ax2.imshow(img_boundary)
    ax2.set_title(f'Explanation (Prediction: {class_names[pred_label]}, {pred_confidence:.2f}%)')
    ax2.axis('off')
    
    # Get feature importance weights
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    
    # Sort features by importance
    features = sorted(dict_heatmap.items(), key=lambda x: np.abs(x[1]), reverse=True)
    
    print("\nTop 5 features that influenced the prediction:")
    for i, (feature, weight) in enumerate(features[:top_features]):
        print(f"Feature {feature}: Weight = {weight:.4f}")
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig('lime_explanation.png')
    
    plt.close()
    
    # Return the explanation and prediction
    return explanation, pred_label

def create_guided_gradcam(model, img_array, layer_name='block5_conv3', class_idx=None, save_path=None):
    """
    Generate Guided Grad-CAM visualization for a model prediction
    
    Args:
        model: The trained model
        img_array: Image to explain
        layer_name: Name of the layer to visualize
        class_idx: Class index to visualize (if None, uses predicted class)
        save_path: Path to save the visualization
        
    Returns:
        guided_gradcam: The guided Grad-CAM visualization
    """
    # Add this function if you want additional visualization methods
    # Implementation would go here
    pass

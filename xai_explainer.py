import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf

def lime_explanation(model, image, class_names, num_samples=1000, top_labels=5):
    """
    Generate LIME explanation for a model prediction
    
    Args:
        model: The trained model
        image: The image to explain
        class_names: List of class names
        num_samples: Number of samples for LIME
        top_labels: Number of top labels to explain
        
    Returns:
        LIME explanation and predicted label
    """
    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Expand dimensions for model prediction
    image_batch = np.expand_dims(image, axis=0)
    
    # Get the model's prediction
    prediction = model.predict(image_batch)[0]
    predicted_class = np.argmax(prediction)
    
    # Define a prediction function for LIME
    def predict_fn(images):
        # Pre-process the images and make predictions
        return model.predict(images)
    
    # Get the explanation
    explanation = explainer.explain_instance(
        image, 
        predict_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples
    )
    
    # Get the mask for the predicted class
    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    # Visualize the explanation
    plt.figure(figsize=(12, 6))
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original Image\nPredicted: {class_names[predicted_class]}")
    plt.axis('off')
    
    # Plot the explanation
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp, mask))
    plt.title('LIME Explanation\nHighlighted regions influenced the prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    plt.close()
    
    # Print detailed explanation
    print(f"Model predicted: {class_names[predicted_class]} with {prediction[predicted_class]*100:.2f}% confidence")
    print("\nTop 5 features that influenced the prediction:")
    
    # Get the explanation for top features
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    
    for feature_id, weight in sorted(dict_heatmap.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Feature {feature_id}: Weight = {weight:.4f}")
    
    return explanation, predicted_class

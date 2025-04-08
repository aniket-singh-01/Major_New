"""
Simple model creator that doesn't rely on complex Lambda layers
Creates a basic CNN that can be used as a fallback model
"""

import tensorflow as tf
import numpy as np
import os

def create_and_save_simple_model(input_shape=(299, 299, 3), num_classes=7):
    """Create and save a simple CNN model for skin lesion classification"""
    print("Creating simple CNN model...")
    
    # Create a simple model with MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create a simple classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model created successfully")
    
    # Save in different formats
    try:
        # Save as Keras model
        keras_path = "simple_cnn_model.keras"
        model.save(keras_path)
        print(f"Model saved to {keras_path}")
        
        # Save as TFLite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        tflite_path = "simple_cnn_model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path}")
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

if __name__ == "__main__":
    create_and_save_simple_model()
    print("\nYou can now use this model with the test scripts:")
    print("- test_model.py")
    print("- derma_test.py")
    print("- simple_tester.py")

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Instead of trying to access layers from InputLayer, use the proper approach
def create_model(input_shape, num_classes):
    # Check if input shape is for image data (has height, width, channels)
    is_image_input = len(input_shape) == 3
    
    # Create appropriate model based on input shape
    if is_image_input:
        # For image input, use a CNN approach
        model = Sequential([
            # Input layer
            Input(shape=input_shape),
            
            # Feature extraction layers
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Flatten the output for the dense layers
            Flatten(),
            # Alternative: GlobalAveragePooling2D(),
            
            # Dense classification layers
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    else:
        # For non-image input (e.g., feature vectors)
        model = Sequential([
            # Don't pass input_shape as a separate argument
            Dense(256, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    
    # Compile with better learning rate and add all metrics
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Increased from 2e-5
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    # Print model summary to verify the architecture
    model.summary()
    
    return model

# Training function with proper callbacks
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """Train model with proper error handling for eager execution"""
    # Ensure eager execution is enabled
    tf.config.run_functions_eagerly(True)
    
    # Convert inputs to tensors to avoid numpy() issues
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        print("Trying alternative training approach...")
        
        # Simpler training approach with smaller batch
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=min(epochs, 20),  # Reduce epochs for faster training
                batch_size=16,  # Smaller batch size
                callbacks=[
                    EarlyStopping(patience=5, restore_best_weights=True)
                ],
                verbose=1
            )
            return history
        except Exception as e2:
            print(f"Alternative training also failed: {e2}")
            # Return empty history to avoid further errors
            class EmptyHistory:
                def __init__(self):
                    self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            
            return EmptyHistory()

# Example usage
# input_shape = (features,)  # Replace with your input shape
# num_classes = 6  # Replace with your number of classes 
# model = create_model(input_shape, num_classes)
# history = train_model(model, X_train, y_train, X_val, y_val)

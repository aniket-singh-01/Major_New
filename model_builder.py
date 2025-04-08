import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def create_model(input_shape, num_classes):
    """
    Create a Sequential model for classification
    
    Args:
        input_shape: Shape of the input data
        num_classes: Number of classes to predict
        
    Returns:
        The compiled model
    """
    # Create a Sequential model
    model = Sequential([
        # Don't use InputLayer directly, use input_shape in the first layer
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

def build_model(base_model, num_classes):
    """
    Build the Inception V3 based model for dermatological diagnosis
    
    Args:
        base_model: The pre-trained Inception V3 model
        num_classes: Number of disease classes to predict
        
    Returns:
        The compiled model
    """
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """
    Train the model with early stopping and learning rate reduction
    
    Args:
        model: The compiled model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        epochs: Maximum number of epochs to train for
        batch_size: Batch size for training
        
    Returns:
        Training history
    """
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
    
    return history

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
import random
from tf_utils import suppress_tf_warnings

class GreyWolfOptimizer:
    def __init__(self, model, X_train, y_train, X_val, y_val, 
                 num_wolves=10, max_iter=10, feature_layer_index=-6):
        """
        Grey Wolf Optimizer for feature selection/optimization
        
        Args:
            model: The trained model to optimize
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            num_wolves: Number of search agents (wolves)
            max_iter: Maximum iterations for optimization
            feature_layer_index: Index of the layer to extract features from
        """
        # Suppress TensorFlow warnings
        suppress_tf_warnings()
        
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.feature_layer_index = feature_layer_index
        
        # Extract features from the specified layer
        self.feature_extractor = Model(
            inputs=model.input,
            outputs=model.layers[feature_layer_index].output
        )
        
        # Get the extracted features
        print("Extracting features from the model...")
        self.train_features = self.feature_extractor.predict(X_train)
        self.val_features = self.feature_extractor.predict(X_val)
        
        # Determine the number of features
        self.num_features = self.train_features.shape[1]
        
    def calculate_fitness(self, wolf_position):
        """
        Calculate fitness for a wolf position (feature subset)
        
        Note: The fitness is negative because we want to maximize accuracy,
        but the optimization algorithm is designed to minimize the objective function.
        Lower (more negative) values indicate better performance.
        """
        # Create mask of selected features
        mask = wolf_position > 0.5
        
        # Skip if no features are selected
        if not np.any(mask):
            return 0  # Poor fitness if no features selected
        
        # Evaluate using the mask
        score = self._evaluate_wolf(mask)
        
        # Return negative accuracy because GWO is minimizing the objective
        # So minimizing -accuracy is equivalent to maximizing accuracy
        return score
    
    def optimize(self):
        """
        Run the Grey Wolf Optimization algorithm
        
        Returns:
            An optimized model
        """
        print(f"Starting Grey Wolf Optimization with {self.num_wolves} wolves for {self.max_iter} iterations")
        print(f"Number of features: {self.num_features}")
        
        # Initialize alpha, beta, and delta positions
        alpha_pos = np.zeros(self.num_features)
        alpha_score = float('inf')
        
        beta_pos = np.zeros(self.num_features)
        beta_score = float('inf')
        
        delta_pos = np.zeros(self.num_features)
        delta_score = float('inf')
        
        # Initialize the positions of wolves (binary position vectors)
        positions = np.random.randint(0, 2, size=(self.num_wolves, self.num_features))
        
        # Main loop
        for iteration in range(self.max_iter):
            for i in range(self.num_wolves):
                # Calculate fitness for this wolf
                fitness = self.calculate_fitness(positions[i])
                
                # Update alpha, beta, and delta
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha_pos = positions[i].copy()
                elif fitness < beta_score:
                    beta_score = fitness
                    beta_pos = positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = positions[i].copy()
            
            # Update the position of each wolf
            a = 2 - iteration * (2 / self.max_iter)  # Parameter decreases linearly from 2 to 0
            
            for i in range(self.num_wolves):
                for j in range(self.num_features):
                    # Calculate coefficients for alpha, beta, and delta
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    # Update position
                    positions[i, j] = (X1 + X2 + X3) / 3
                
                # Binarize positions
                positions[i] = np.where(positions[i] > 0.5, 1, 0)
            
            # Display absolute accuracy value to avoid confusion
            actual_accuracy = -alpha_score
            print(f"Iteration {iteration+1}/{self.max_iter}, " 
                  f"Best fitness: {alpha_score} (Accuracy: {actual_accuracy:.4f})")
        
        # Create final model using alpha (best) features
        print("Optimization complete. Creating final model with selected features.")
        final_model = self._create_model_with_features(alpha_pos > 0.5)
        return final_model
    
    def _evaluate_wolf(self, feature_mask):
        """
        Evaluate a wolf (feature subset) by building and testing a model
        
        Args:
            feature_mask: Boolean mask of selected features
            
        Returns:
            Fitness score (lower is better)
        """
        # Create a simple model to evaluate this feature subset
        input_shape = (sum(feature_mask),)
        
        # Use TensorFlow 2.x API style
        inputs = Input(shape=input_shape)
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        outputs = Dense(self.y_train.shape[1], activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train on selected features
        X_train_selected = self.train_features[:, feature_mask]
        X_val_selected = self.val_features[:, feature_mask]
        
        # Use smaller epochs and batch size for faster evaluation
        model.fit(
            X_train_selected, self.y_train,
            epochs=3,  # Reduce from 5 to 3 for faster evaluation
            batch_size=32,
            validation_data=(X_val_selected, self.y_val),
            verbose=0
        )
        
        # Evaluate
        _, val_acc = model.evaluate(X_val_selected, self.y_val, verbose=0)
        
        # Fitness is negative accuracy (to minimize) plus a penalty for using too many features
        feature_ratio = sum(feature_mask) / self.num_features
        fitness = -val_acc + 0.1 * feature_ratio
        
        return fitness
    
    def _create_model_with_features(self, feature_mask):
        """
        Create a final model using the selected features
        
        Args:
            feature_mask: Boolean mask of selected features
            
        Returns:
            Optimized model
        """
        # Create a new feature extractor that outputs only selected features
        selected_features = sum(feature_mask)
        print(f"Creating model with {selected_features} selected features out of {self.num_features}")
        
        # Use TensorFlow 2.x API style with no deprecated functions
        # Create the final model structure using functional API
        inputs = self.model.input
        x = self.feature_extractor(inputs)
        
        # Apply feature mask using a custom Lambda layer - modern approach in TF 2.x
        feature_mask_tf = tf.constant(feature_mask.astype('float32'))
        
        # Add output_shape parameter to Lambda layer to fix the issue
        mask_layer = tf.keras.layers.Lambda(
            lambda x: tf.multiply(x, feature_mask_tf),
            output_shape=lambda input_shape: input_shape  # Explicitly define output shape
        )(x)
        
        # Continue with dense layers as in the original model
        x = Dense(1024, activation='relu')(mask_layer)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.y_train.shape[1], activation='softmax')(x)
        
        optimized_model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the optimized model
        optimized_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the optimized model
        optimized_model.fit(
            self.X_train, self.y_train,
            epochs=10,  # Reduced from 15 for faster training
            batch_size=32,
            validation_data=(self.X_val, self.y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ],
            verbose=1
        )
        
        # Save the model in TFLite format as well for better compatibility
        try:
            # Convert the model to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(optimized_model)
            tflite_model = converter.convert()
            
            # Save the TFLite model
            with open('optimized_model.tflite', 'wb') as f:
                f.write(tflite_model)
            
            print("Model also saved in TFLite format for better compatibility")
        except Exception as e:
            print(f"Could not save TFLite model: {e}")
        
        # Also save in saved_model format for maximum compatibility
        try:
            tf.saved_model.save(optimized_model, 'saved_model')
            print("Model also saved in SavedModel format")
        except Exception as e:
            print(f"Could not save in SavedModel format: {e}")
        
        return optimized_model

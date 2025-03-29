import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
import random

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
        self.train_features = self.feature_extractor.predict(X_train)
        self.val_features = self.feature_extractor.predict(X_val)
        
        # Determine the number of features
        self.num_features = self.train_features.shape[1]
        
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
                # Create a mask with 1s where features are selected
                mask = positions[i] > 0.5
                
                # Skip if no features are selected
                if not np.any(mask):
                    continue
                
                # Create a new model with selected features
                score = self._evaluate_wolf(mask)
                
                # Update alpha, beta, and delta
                if score < alpha_score:
                    alpha_score = score
                    alpha_pos = positions[i].copy()
                elif score < beta_score:
                    beta_score = score
                    beta_pos = positions[i].copy()
                elif score < delta_score:
                    delta_score = score
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
            
            print(f"Iteration {iteration+1}/{self.max_iter}, Best fitness: {alpha_score}")
        
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
        
        inputs = Input(shape=input_shape)
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        outputs = Dense(self.y_train.shape[1], activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train on selected features
        X_train_selected = self.train_features[:, feature_mask]
        X_val_selected = self.val_features[:, feature_mask]
        
        model.fit(
            X_train_selected, self.y_train,
            epochs=5,
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
        
        # Create the final model structure
        inputs = self.model.input
        x = self.feature_extractor(inputs)
        
        # Apply feature mask to keep only selected features
        # This is a custom layer that multiplies the features by the mask
        mask_layer = tf.keras.layers.Lambda(
            lambda x: tf.multiply(x, tf.constant(feature_mask.astype('float32')))
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
            epochs=15,
            batch_size=32,
            validation_data=(self.X_val, self.y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        return optimized_model

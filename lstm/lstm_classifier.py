import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMLogClassifier:
    """LSTM model for classifying log importance"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def build_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Build model: LSTM for sequential input, MLP for tabular/TF-IDF input"""
        logger.info(f"Building model with input shape: {input_shape}")
        
        # Use functional API for better compatibility
        inputs = keras.Input(shape=input_shape)
        
        if len(input_shape) == 2:
            # Sequential input: (timesteps, features) → use LSTM
            logger.info("Using LSTM layers for sequential input")
            x = layers.LSTM(self.config['lstm_units'], return_sequences=True, dropout=self.config['dropout_rate'])(inputs)
            x = layers.LSTM(self.config['lstm_units'] // 2, dropout=self.config['dropout_rate'])(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
        else:
            # Tabular/TF-IDF input: (features,) → use MLP
            logger.info("Using MLP layers for tabular/TF-IDF input")
            x = layers.Dense(self.config['embedding_dim'], activation='relu')(inputs)
            x = layers.Dropout(self.config['dropout_rate'])(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info("Model built successfully (LSTM for sequence, MLP for tabular)")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the LSTM model"""
        logger.info("Starting LSTM model training...")
        logger.info(f"Input data shape: {X.shape}")
        
        # Ensure data is 2D for tabular features
        if len(X.shape) == 3:
            # If somehow we have 3D data, flatten the last two dimensions
            X = X.reshape(X.shape[0], -1)
            logger.info(f"Reshaped input data to: {X.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config['validation_split'], 
            random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        
        # Build model
        self.model = self.build_model(X_train.shape[1:])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_metrics = self.model.evaluate(X_train, y_train, verbose=0)
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Extract metrics (loss, accuracy, precision, recall)
        train_loss, train_acc, train_prec, train_rec = train_metrics
        val_loss, val_acc, val_prec, val_rec = val_metrics
        
        # Generate predictions for detailed metrics
        y_pred = (self.model.predict(X_val) > 0.5).astype(int).flatten()
        
        results = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_precision': train_prec,
            'val_precision': val_prec,
            'train_recall': train_rec,
            'val_recall': val_rec,
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        logger.info(f"Training completed. Validation accuracy: {val_acc:.4f}")
        logger.info("Classification Report:")
        logger.info(results['classification_report'])
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Ensure data is 2D for tabular features
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions"""
        predictions = self.predict(X)
        return (predictions > threshold).astype(int)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        # Always use .keras extension
        if not filepath.endswith('.keras'):
            filepath = filepath.rsplit('.', 1)[0] + '.keras'
        self.model.save(filepath, save_format='keras')
        # Save config and preprocessing info
        model_info = {
            'config': self.config,
            'input_shape': self.model.input_shape[1:]
        }
        joblib.dump(model_info, filepath.replace('.keras', '_info.pkl'))
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        # Always use .keras extension
        if not filepath.endswith('.keras'):
            filepath = filepath.rsplit('.', 1)[0] + '.keras'
        self.model = keras.models.load_model(filepath)
        # Load config and preprocessing info
        try:
            model_info = joblib.load(filepath.replace('.keras', '_info.pkl'))
            self.config = model_info['config']
        except FileNotFoundError:
            logger.warning("Model info file not found. Using default config.")
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, X: np.ndarray, feature_names: Optional[list] = None) -> dict:
        """Get feature importance using permutation importance"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Ensure data is 2D for tabular features
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        baseline_predictions = self.predict(X)
        baseline_score = np.mean(baseline_predictions)
        
        importance_scores = {}
        
        for i in range(X.shape[-1]):
            # Create permuted version
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get predictions with permuted feature
            permuted_predictions = self.predict(X_permuted)
            permuted_score = np.mean(permuted_predictions)
            
            # Calculate importance as difference in predictions
            importance = abs(baseline_score - permuted_score)
            
            feature_name = feature_names[i] if feature_names else f"feature_{i}"
            importance_scores[feature_name] = importance
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance
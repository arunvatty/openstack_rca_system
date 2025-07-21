import numpy as np
import pandas as pd
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
        """Build LSTM model architecture"""
        logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # Embedding layer (if needed)
            layers.Dense(self.config['embedding_dim'], activation='relu'),
            layers.Dropout(self.config['dropout_rate']),
            
            # LSTM layers
            layers.LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate']
            ),
            layers.LSTM(
                self.config['lstm_units'] // 2,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate']
            ),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.config['dropout_rate']),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.config['dropout_rate']),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("LSTM model built successfully")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the LSTM model"""
        logger.info("Starting LSTM model training...")
        
        # Reshape data for LSTM if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
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
        train_loss, train_acc, train_prec, train_rec = self.model.evaluate(
            X_train, y_train, verbose=0
        )
        val_loss, val_acc, val_prec, val_rec = self.model.evaluate(
            X_val, y_val, verbose=0
        )
        
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
        
        # Reshape data for LSTM if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
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
        
        self.model.save(filepath)
        
        # Save config and preprocessing info
        model_info = {
            'config': self.config,
            'input_shape': self.model.input_shape[1:]
        }
        joblib.dump(model_info, filepath.replace('.h5', '_info.pkl'))
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Load config and preprocessing info
        try:
            model_info = joblib.load(filepath.replace('.h5', '_info.pkl'))
            self.config = model_info['config']
        except FileNotFoundError:
            logger.warning("Model info file not found. Using default config.")
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, X: np.ndarray, feature_names: Optional[list] = None) -> dict:
        """Get feature importance using permutation importance"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Reshape data for LSTM if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        baseline_predictions = self.predict(X)
        baseline_score = np.mean(baseline_predictions)
        
        importance_scores = {}
        
        for i in range(X.shape[-1]):
            # Create permuted version
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, :, i])
            
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
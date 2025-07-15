"""
Machine learning models for genre classification.
Includes traditional ML models and deep learning approaches.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any, Optional
import time

# Traditional ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep Learning
import tensorflow as tf
import keras
from keras import layers, models
from keras.utils import to_categorical

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

class GenreClassifier:
    """Base class for genre classification models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GenreClassifier':
        """Fit the model to the data."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted,
            'model_name': self.model_name
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.is_fitted = data['is_fitted']
        self.model_name = data['model_name']
        print(f"Model loaded from {filepath}")


class RandomForestGenreClassifier(GenreClassifier):
    """Random Forest classifier for genre classification."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestGenreClassifier':
        """Fit the Random Forest model."""
        print("Fitting Random Forest classifier")
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit model
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        training_time = time.time() - start_time
        print(f"Random Forest training completed in {training_time:.2f} seconds")
        
        return self


class SVMClassifier(GenreClassifier):
    """Support Vector Machine classifier for genre classification."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, random_state: int = 42):
        super().__init__("SVM")
        self.model = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            probability=True
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Fit the SVM model."""
        print("Fitting SVM classifier")
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit model
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        training_time = time.time() - start_time
        print(f"SVM training completed in {training_time:.2f} seconds")
        
        return self


class CNNClassifier(GenreClassifier):
    """Convolutional Neural Network for genre classification using spectrograms."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 1), num_classes: int = 10):
        super().__init__("CNN")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:

        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation= 'relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation ='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32, 
            validation_split: float = 0.2) -> 'CNNClassifier':
        
        print("Fitting CNN classifier:")
        start_time = time.time()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=self.num_classes)
        
        # Reshape input for CNN if needed
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])), 1)
        else:
            X_reshaped = X
        
        # Fit model
        self.model.fit(
            X_reshaped, y_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        self.is_fitted = True
        
        training_time = time.time() - start_time
        print(f"CNN training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with CNN."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Reshape input for CNN
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])), 1)
        else:
            X_reshaped = X
        
        predictions = self.model.predict(X_reshaped)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))


class XGBoostClassifier(GenreClassifier):
    """XGBoost classifier for genre classification."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__("XGBoost")
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='mlogloss'
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifier':
        """Fit the XGBoost model."""
        print("Fitting XGBoost classifier")
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit model
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        training_time = time.time() - start_time
        print(f"XGBoost training completed in {training_time:.2f} seconds")
        
        return self


class ModelEnsemble:
    """Ensemble of multiple models for genre classification."""
    
    def __init__(self, models: List[GenreClassifier]):
        self.models = models
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelEnsemble':
        """Fit all models in the ensemble."""
        print("Training ensemble models...")
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, method: str = 'voting') -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if method == 'voting':
            # Simple voting
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            
            # Take majority vote
            predictions_array = np.array(predictions)
            ensemble_pred = []
            for i in range(len(X)):
                votes = predictions_array[:, i]
                # Get most common prediction
                unique, counts = np.unique(votes, return_counts=True)
                ensemble_pred.append(unique[np.argmax(counts)])
            
            return np.array(ensemble_pred)
        
        elif method == 'averaging':
            # Average probabilities
            probas = []
            for model in self.models:
                proba = model.predict_proba(X)
                probas.append(proba)
            
            avg_proba = np.mean(probas, axis=0)
            return np.argmax(avg_proba, axis=1)
        
        else:
            raise ValueError("Method must be 'voting' or 'averaging'")


class MLPGenreClassifier(GenreClassifier):
    """Feedforward neural network (MLP) for genre classification using tabular features."""
    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__("MLP")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 32, validation_split: float = 0.2):
        print("Fitting MLP classifier:")
        start_time = time.time()
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=self.num_classes)
        self.model.fit(X_scaled, y_categorical, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        self.is_fitted = True
        training_time = time.time() - start_time
        print(f"MLP training completed in {training_time:.2f} seconds")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))


def create_models() -> Dict[str, GenreClassifier]:
    """Create a dictionary of models to train and compare."""
    models = {
        'random_forest': RandomForestGenreClassifier(n_estimators=100, max_depth=10),
        'svm_rbf': SVMClassifier(kernel='rbf', C=1.0),
        'svm_linear': SVMClassifier(kernel='linear', C=1.0),
    }
    
    # Add XGBoost if avail
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBoostClassifier(n_estimators=100, max_depth=6)
    
    # Add CNN (simplified version)
    models['cnn'] = CNNClassifier()
    
    return models


def train_all_models(X: np.ndarray, y: np.ndarray, models: Dict[str, GenreClassifier]) -> Dict[str, GenreClassifier]:
    """Train all models and return the fitted models."""
    fitted_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name.upper()} model")
        print(f"{'='*50}")
        
        try:
            fitted_model = model.fit(X, y)
            fitted_models[name] = fitted_model
            print(f"✓ {name} training completed successfully")
        except Exception as e:
            print(f"✗ {name} training failed: {str(e)}")
    
    return fitted_models


def evaluate_models(models: Dict[str, GenreClassifier], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Evaluate all models and return performance metrics."""
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            print(f"✓ {name} - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"✗ {name} evaluation failed: {str(e)}")
            results[name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    return results


def save_models(models: Dict[str, GenreClassifier], save_dir: str = "models/trained"):
    """Save all trained models."""
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        try:
            filepath = os.path.join(save_dir, f"{name}_model.pkl")
            model.save(filepath)
        except Exception as e:
            print(f"Failed to save {name}: {str(e)}")


def load_best_model(models_dir: str = "models/trained") -> Optional[GenreClassifier]:
    """Load the best performing model."""
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            model_path = os.path.join(models_dir, model_files[0])

            dummy_model = RandomForestGenreClassifier()
            dummy_model.load(model_path)
            return dummy_model
    
    return None


def train_model(X_train, y_train):
    """
    Train a Random Forest model
    """
    model = RandomForestGenreClassifier(
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model
    Returns:
        Tuple of (accuracy, classification_report, confusion_matrix)
    """
    # Make predictions
    y_pred_raw = model.model.predict(X_test)
    # If output is probabilities (Keras), convert to class indices
    if len(y_pred_raw.shape) > 1 and y_pred_raw.shape[1] > 1:
        y_pred_num = np.argmax(y_pred_raw, axis=1)
    else:
        y_pred_num = y_pred_raw
    # Decode predictions to string labels if label_encoder exists
    if hasattr(model, 'label_encoder') and hasattr(model.label_encoder, 'inverse_transform'):
        y_pred = model.label_encoder.inverse_transform(y_pred_num)
    else:
        y_pred = y_pred_num
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm


def save_model(model, filepath):

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    return joblib.load(filepath) 
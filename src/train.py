"""
Training script for music genre classification models.
Handles data loading, model training, and evaluation.
"""

import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import load_and_prepare_data
from src.models import train_model, evaluate_model, save_model

def main():
    print("Music Genre Classifier Training")
    
    # Load and prepare data (auto-detects best available)
    print("\n1. Loading and preparing data")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    # Train model
    print("\n2. Training Random Forest model")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\n3. Evaluating model...")
    accuracy, report, cm = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save model and scaler
    print("\n4. Saving model and scaler")
    save_model(model, "models/random_forest_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Create and save confusion matrix plot
    print("\n5. Creating confusion matrix plot")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y_test.unique()), 
                yticklabels=sorted(y_test.unique()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    
    print("\nTraining Complete")
    print(f"Model saved to: models/random_forest_model.pkl")
    print(f"Scaler saved to: models/scaler.pkl")
    print(f"Confusion matrix saved to: models/confusion_matrix.png")

if __name__ == "__main__":
    main() 
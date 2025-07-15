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

def main(model_type="random_forest"):
    print("Music Genre Classifier Training")
    
    # Load and prepare data (auto-detects best available)
    print("\n1. Loading and preparing data")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    # Train model
    if model_type == "cnn":
        print("\n2. Training CNN model")
        from src.models import CNNClassifier
        num_classes = len(np.unique(y_train))
        input_shape = (int(np.sqrt(X_train.shape[1])), int(np.sqrt(X_train.shape[1])), 1)
        cnn = CNNClassifier(input_shape=input_shape, num_classes=num_classes)
        model = cnn.fit(X_train, y_train, epochs=20, batch_size=32)
    elif model_type == "mlp":
        print("\n2. Training MLP (Neural Network) model")
        from src.models import MLPGenreClassifier
        num_classes = len(np.unique(y_train))
        mlp = MLPGenreClassifier(input_dim=X_train.shape[1], num_classes=num_classes)
        model = mlp.fit(X_train, y_train, epochs=20, batch_size=32)
        # Ensure y_train and y_test are 1D arrays of class labels for evaluation
        if y_train.ndim > 1:
            y_train = np.argmax(y_train, axis=1)
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)
    else:
        print("\n2. Training Random Forest model")
        from src.models import train_model
        model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\n3. Evaluating model...")
    from src.models import evaluate_model
    accuracy, report, cm = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save model and scaler
    print("\n4. Saving model and scaler")
    if model_type == "cnn":
        model.model.save("models/cnn_model.h5")
        import joblib
        joblib.dump(model.label_encoder, "models/cnn_label_encoder.pkl")
        joblib.dump(scaler, "models/cnn_scaler.pkl")
    elif model_type == "mlp":
        model.model.save("models/mlp_model.h5")
        import joblib
        joblib.dump(model.label_encoder, "models/mlp_label_encoder.pkl")
        joblib.dump(scaler, "models/mlp_scaler.pkl")
    else:
        from src.models import save_model
        save_model(model, "models/random_forest_model.pkl")
        import joblib
        joblib.dump(scaler, "models/scaler.pkl")
    
    # Create and save confusion matrix plot
    print("\n5. Creating confusion matrix plot")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    import pandas as pd
    y_labels = sorted(list(set(list(y_test) + list(y_train))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=y_labels, 
                yticklabels=y_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if model_type == "cnn":
        plt.savefig('models/confusion_matrix_cnn.png', dpi=300, bbox_inches = 'tight')
    elif model_type == "mlp":
        plt.savefig('models/confusion_matrix_mlp.png', dpi=300, bbox_inches = 'tight')
    else:
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    
    print("\nTraining Complete")
    if model_type == "cnn":
        print(f"Model saved to: models/cnn_model.h5")
        print(f"Scaler saved to: models/cnn_scaler.pkl")
        print(f"Label encoder saved to: models/cnn_label_encoder.pkl")
        print(f"Confusion matrix saved to: models/confusion_matrix_cnn.png")
    elif model_type == "mlp":
        print(f"Model saved to: models/mlp_model.h5")
        print(f"Scaler saved to: models/mlp_scaler.pkl")
        print(f"Label encoder saved to: models/mlp_label_encoder.pkl")
        print(f"Confusion matrix saved to: models/confusion_matrix_mlp.png")
    else:
        print(f"Model saved to: models/random_forest_model.pkl")
        print(f"Scaler saved to: models/scaler.pkl")
        print(f"Confusion matrix saved to: models/confusion_matrix.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main() 
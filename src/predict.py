import joblib
import numpy as np
import pandas as pd
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import load_model

def predict_genre(features, model_path="models/random_forest_model.pkl", scaler_path="models/scaler.pkl"):

    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Ensure features is 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction using the model's internal model
    y_pred_num = model.model.predict(features_scaled)
    
    # Decode predictions to string labels
    if hasattr(model, 'label_encoder') and hasattr(model.label_encoder, 'inverse_transform'):
        prediction = model.label_encoder.inverse_transform(y_pred_num)[0]
    else:
        prediction = y_pred_num[0]
    
    return prediction

def predict_from_csv(csv_path, model_path="models/random_forest_model.pkl", scaler_path="models/scaler.pkl"):

    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract features (exclude filename, length, and label columns if present)
    feature_columns = [col for col in df.columns if col not in ['filename', 'length', 'label', 'genre']]
    features = df[feature_columns].values
    
    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions using the model's internal model
    y_pred_num = model.model.predict(features_scaled)
    
    # Decode predictions to string labels
    if hasattr(model, 'label_encoder') and hasattr(model.label_encoder, 'inverse_transform'):
        predictions = model.label_encoder.inverse_transform(y_pred_num)
    else:
        predictions = y_pred_num
    
    # Add predictions to dataframe
    df['predicted_genre'] = predictions
    
    return df

def main():

    print("Music Genre Prediction")
    
    # Check if model exists
    model_path = "models/random_forest_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train the model first using: python src/train.py")
        return
    
    # Example: Predict from the test data
    test_data_path = "data/raw/Data/features_30_sec.csv"
    if os.path.exists(test_data_path):
        print(f"\nMaking predictions on {test_data_path}")
        
        # Load a small sample for demonstration
        df = pd.read_csv(test_data_path)
        sample_df = df.head(5)  # Take first 5 samples
        
        # Save sample to temporary file
        sample_path = "temp_sample.csv"
        sample_df.to_csv(sample_path, index=False)
        
        # Make predictions
        results = predict_from_csv(sample_path, model_path)
        
        # Display results
        print("\nPrediction Results:")
        print("=" * 50)
        for i, row in results.iterrows():
            actual = row.get('label', 'Unknown')
            predicted = row['predicted_genre']
            filename = row.get('filename', f'Sample {i+1}')
            
            status = "✓" if actual == predicted else "✗"
            print(f"{status} {filename}")
            print(f"   Actual: {actual}")
            print(f"   Predicted: {predicted}")
            print()
        
        # Clean up
        os.remove(sample_path)
        
        # Calculate accuracy if labels are available
        if 'label' in results.columns:
            accuracy = (results['label'] == results['predicted_genre']).mean()
            print(f"Sample Accuracy: {accuracy:.2%}")
    
    else:
        print(f"Test data not found at {test_data_path}")

if __name__ == "__main__":
    main() 
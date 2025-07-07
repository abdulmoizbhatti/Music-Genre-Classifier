import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

"""
Unified function to load features from any CSV file (pre-extracted or custom-extracted)
    data_path: Path to the CSV file
    X: Feature matrix
    y: Target labels
"""

def load_features(data_path):

    print(f"Loading features from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Handle different label column names
    if 'label' in df.columns:
        y = df['label']
        print("Using 'label' column for targets")
    elif 'genre' in df.columns:
        y = df['genre']
        print("Using 'genre' column for targets")
    else:
        raise ValueError("No 'label' or 'genre' column found in the CSV file")
    
    # Remove non-feature columns automatically
    non_feature_cols = ['filename', 'length', 'label', 'genre']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    X = df[feature_cols]
    
    print(f"Loaded {len(df)} samples with {X.shape[1]} features")
    print(f"Genres: {y.unique()}")
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

"""
Main function to load and prepare data with automatic detection
    data_path: Optional path to specific CSV file. If None, auto-detects best available data.
"""

def load_and_prepare_data(data_path=None):

    if data_path is None:
        # Auto-detect best available data
        if os.path.exists("data/processed/features.csv"):
            data_path = "data/processed/features.csv"
            print("Using custom-extracted features")
        elif os.path.exists("data/raw/Data/features_30_sec.csv"):
            data_path = "data/raw/Data/features_30_sec.csv"
            print("Using pre-extracted features")
        else:
            raise FileNotFoundError("No feature files found. Please run feature extraction or ensure pre-extracted data is available.")
    
    # Load features
    X, y = load_features(data_path)
    
    # Prepare data for training
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    return X_train, X_test, y_train, y_test, scaler

# Keep the old function for backward compatibility
def load_pre_extracted_features(data_path):
    return load_features(data_path) 
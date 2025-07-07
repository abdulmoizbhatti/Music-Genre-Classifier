#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.feature_extraction import extract_features_for_dataset
from src.train import main as train_models


def setup_environment():
    """Create necessary directories for the project."""
    directories = [
        "data/raw",
        "data/processed", 
        "models/trained",
        "models/results",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def run_feature_extraction(data_dir: str = "data/raw", output_path: str = "data/processed/features.csv"):
    """Run feature extraction on audio files."""
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        return False
    
    # Check if audio files exist
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"No audio files found in {data_dir}")
        return False
    
    print(f"Found {len(audio_files)} audio files")
    
    # Extract features
    try:
        features_df, labels = extract_features_for_dataset(data_dir, output_path)
        if features_df is not None:
            print("Feature extraction completed successfully!")
            return True
        else:
            print("Feature extraction failed!")
            return False
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return False


def run_training():
    """Run the model training pipeline using unified data processing."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    try:
        train_models()
        print("Model training completed successfully!")
        return True
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description= "Music Genre Classifier")
    parser.add_argument("--extract", action="store_true", help= "Extract features from audio files")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--full", action = "store_true", help="Run complete pipeline")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing audio files")
    
    args = parser.parse_args()
    
    print("Music Genre Classifier")
    print("="*60)
    
    # Setup environment
    setup_environment()
    
    # Run feature extraction
    if args.extract or args.full:
        success = run_feature_extraction(args.data_dir)
        if not success:
            print("\nFeature extraction failed. Please check your data directory.")
            return
    
    # Run training
    if args.train or args.full:
        success = run_training()
        if not success:
            print("\nTraining failed. Check your data files.")
            return
    
    # If no specific action, show help
    if not any([args.extract, args.train, args.full]):
        print("\nUsage:")
        print("  python main.py --extract   # Extract features from audio files")
        print("  python main.py --train     # Train models")
        print("  python main.py --full      # Run complete pipeline")
        print("\nExample:")
        print("  python main.py --full      # Run everything")
        
        # Check what data is available
        if os.path.exists("data/processed/features.csv"):
            print("\n✓ Custom-extracted features found. You can run training with:")
            print("  python main.py --train")
        elif os.path.exists("data/raw/Data/features_30_sec.csv"):
            print("\n✓ Pre-extracted features found. You can run training with:")
            print("  python main.py --train")
        else:
            print("\nNo features found. Please add audio files and run:")
            print("  python main.py --extract")


if __name__ == "__main__":
    main() 
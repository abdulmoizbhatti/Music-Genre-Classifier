"""
Audio feature extraction module for music genre classification.
Extracts various audio features using librosa library.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
import joblib
from typing import List, Tuple, Dict, Optional


class AudioFeatureExtractor:
    
    def __init__(self, sample_rate: int = 22050, duration: float = 30.0):
        """
        sample_rate: Target sample rate for audio processing duration: Duration of audio segments to process (seconds)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_names = []
        
    def extract_features_from_file(self, file_path: str) -> Dict[str, float]:
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Extract features
            features = {}
            
            # Mel-frequency cepstral coefficients
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
            features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                features[f'chroma_{i}_std'] = np.std(chroma[i])
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(7):
                features[f'contrast_{i}_mean'] = np.mean(contrast[i])
                features[f'contrast_{i}_std'] = np.std(contrast[i])
            
            # Ensure all features are single floats (not arrays or lists)
            for k, v in features.items():
                if isinstance(v, (np.ndarray, list)):
                    if np.size(v) == 1:
                        features[k] = float(np.ravel(v)[0])
                    else:
                        features[k] = float(np.mean(v))
                else:
                    features[k] = float(v)

            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return {}
    
    def extract_features_from_directory(self, data_dir: str, labels: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, List[str]]:

        features_list = []
        genre_labels = []
        
        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        # Get all audio files
        audio_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Extract features from each file
        for file_path in tqdm(audio_files, desc="Extracting features"):
            features = self.extract_features_from_file(file_path)
            
            if features:  # Only add if features were successfully extracted
                features_list.append(features)
                
                # Determine genre label
                if labels and file_path in labels:
                    genre_labels.append(labels[file_path])
                else:
                    # Try to extract genre from file path or directory structure
                    genre = self._extract_genre_from_path(file_path)
                    genre_labels.append(genre)
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Store feature names for later use
        self.feature_names = list(features_df.columns)
        
        return features_df, genre_labels
    
    def _extract_genre_from_path(self, file_path: str) -> str:

        # Common genre patterns in file paths
        path_lower = file_path.lower()
        
        # Define genre keywords
        genre_keywords = {
            'blues': ['blues'],
            'classical': ['classical', 'classic'],
            'country': ['country'],
            'disco': ['disco'],
            'hiphop': ['hiphop', 'hip-hop', 'rap'],
            'jazz': ['jazz'],
            'metal': ['metal'],
            'pop': ['pop'],
            'reggae': ['reggae'],
            'rock': ['rock']
        }
        
        # Check for genre keywords in path
        for genre, keywords in genre_keywords.items():
            if any(keyword in path_lower for keyword in keywords):
                return genre
        
        # If no genre found, try to extract from directory structure
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            part_lower = part.lower()
            for genre, keywords in genre_keywords.items():
                if any(keyword in part_lower for keyword in keywords):
                    return genre
        
        # Default to 'unknown' if no genre found
        return 'unknown'
    
    def save_features(self, features_df: pd.DataFrame, labels: List[str], output_path: str):
        # Add labels to DataFrame
        features_df['genre'] = labels
        
        # Save to CSV
        features_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
    
    def load_features(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:

        df = pd.read_csv(file_path)
        labels = df['genre'].tolist()
        features_df = df.drop('genre', axis=1)
        
        # Update feature names
        self.feature_names = list(features_df.columns)
        
        return features_df, labels


def extract_features_for_dataset(data_dir: str, output_path: str = "data/processed/features.csv"):

    extractor = AudioFeatureExtractor()
    
    print("Extracting features from audio files...")
    features_df, labels = extractor.extract_features_from_directory(data_dir)
    
    if not features_df.empty:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save features
        extractor.save_features(features_df, labels, output_path)
        
        print(f"Extracted {len(features_df)} samples with {len(features_df.columns)} features")
        print(f"Genres found: {set(labels)}")
        
        return features_df, labels
    else:
        print("No features were extracted. Please check your data directory.")
        return None, None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/features.csv"
        
        features_df, labels = extract_features_for_dataset(data_dir, output_path)
        
        if features_df is not None:
            print("\nFeature extraction completed successfully!")
            print(f"Dataset shape: {features_df.shape}")
            print(f"Number of genres: {len(set(labels))}")
    else:
        print("Usage: python feature_extraction.py <data_directory> [output_path]") 
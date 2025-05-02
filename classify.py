import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Import our configuration and utilities
from config_manager import ConfigManager, get_config
from utilities import VideoUtilities, DataUtilities

"""
Classification module for Thirdeye deepfake detection system.
Provides functionality for classifying videos as deepfake or real.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thirdeye_classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Classification")


class ClassificationResults:
    """Class for handling and visualizing classification results."""
    
    def __init__(self, predictions: Dict[str, Dict[str, float]], threshold: float = 0.5):
        """
        Initialize the classification results.
        
        Args:
            predictions: Dictionary of predictions {filename: {'Real': prob, 'Deepfake': prob}}
            threshold: Classification threshold (default: 0.5)
        """
        self.predictions = predictions
        self.threshold = threshold
        
        # Calculate labels based on threshold
        self.labels = {}
        for filename, probs in predictions.items():
            if probs['Real'] >= threshold:
                self.labels[filename] = 'Real'
            else:
                self.labels[filename] = 'Deepfake'
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.
        
        Returns:
            DataFrame with predictions
        """
        data = []
        
        for filename, probs in self.predictions.items():
            data.append({
                'Filename': filename,
                'Real_Probability': probs['Real'],
                'Deepfake_Probability': probs['Deepfake'],
                'Predicted_Class': self.labels[filename]
            })
        
        return pd.DataFrame(data)
    
    def save_to_csv(self, output_path: str) -> None:
        """
        Save results to a CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Classification results saved to {output_path}")
    
    def print_results(self) -> None:
        """Print the classification results to the console."""
        df = self.to_dataframe()
        
        for _, row in df.iterrows():
            filename = row['Filename']
            real_prob = row['Real_Probability'] * 100
            df_prob = row['Deepfake_Probability'] * 100
            predicted = row['Predicted_Class']
            
            print(f"========== Video {filename} ==========")
            print(f"Real: {real_prob:.2f}%, Deepfake: {df_prob:.2f}%")
            print(f"{filename}: {predicted}\n")
    
    def plot_confidence_histogram(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a histogram of confidence scores.
        
        Args:
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        df = self.to_dataframe()
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of real probabilities
        sns.histplot(df['Real_Probability'], bins=20, 
                   label='Real Confidence', alpha=0.7, ax=ax)
        
        # Plot histogram of deepfake probabilities
        sns.histplot(df['Deepfake_Probability'], bins=20, 
                   label='Deepfake Confidence', alpha=0.7, ax=ax)
        
        # Add threshold line
        plt.axvline(x=self.threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.threshold})')
        
        # Set labels and title
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Videos')
        plt.title('Confidence Score Distribution')
        plt.legend()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence histogram saved to {output_path}")
        
        return fig
    
    def plot_top_predictions(self, n: int = 5, 
                           output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot top confident predictions for both classes.
        
        Args:
            n: Number of top predictions to show
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        df = self.to_dataframe()
        
        # Get top real and deepfake predictions
        top_real = df.sort_values('Real_Probability', ascending=False).head(n)
        top_deepfake = df.sort_values('Deepfake_Probability', ascending=False).head(n)
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot top real predictions
        ax1.barh(top_real['Filename'], top_real['Real_Probability'])
        ax1.set_title(f'Top {n} Real Predictions')
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Confidence Score')
        
        # Plot top deepfake predictions
        ax2.barh(top_deepfake['Filename'], top_deepfake['Deepfake_Probability'])
        ax2.set_title(f'Top {n} Deepfake Predictions')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Confidence Score')
        
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Top predictions plot saved to {output_path}")
        
        return fig


class PredictionAggregator:
    """Class for aggregating predictions from multiple video segments."""
    
    def __init__(self, strategy: str = 'mean'):
        """
        Initialize the prediction aggregator.
        
        Args:
            strategy: Aggregation strategy ('mean', 'median', 'max', 'voting')
        """
        self.strategy = strategy
        
        # Map strategy to aggregation function
        self.aggregation_functions = {
            'mean': self._mean_aggregation,
            'median': self._median_aggregation,
            'max': self._max_aggregation,
            'voting': self._voting_aggregation,
            'weighted': self._weighted_aggregation
        }
        
        if strategy not in self.aggregation_functions:
            raise ValueError(f"Unsupported aggregation strategy: {strategy}. "
                           f"Available strategies: {list(self.aggregation_functions.keys())}")
    
    def aggregate(self, predictions: np.ndarray, segment_info: Optional[Dict] = None) -> np.ndarray:
        """
        Aggregate predictions from multiple segments.
        
        Args:
            predictions: Array of predictions (num_segments, num_classes)
            segment_info: Optional information about segments (quality, confidence, etc.)
            
        Returns:
            Aggregated prediction (num_classes,)
        """
        if predictions.shape[0] == 0:
            raise ValueError("No predictions to aggregate")
        
        if predictions.shape[0] == 1:
            return predictions[0]
        
        return self.aggregation_functions[self.strategy](predictions, segment_info)
    
    def _mean_aggregation(self, predictions: np.ndarray, 
                         segment_info: Optional[Dict] = None) -> np.ndarray:
        """Simple mean aggregation."""
        return np.mean(predictions, axis=0)
    
    def _median_aggregation(self, predictions: np.ndarray, 
                           segment_info: Optional[Dict] = None) -> np.ndarray:
        """Median aggregation (robust to outliers)."""
        return np.median(predictions, axis=0)
    
    def _max_aggregation(self, predictions: np.ndarray, 
                        segment_info: Optional[Dict] = None) -> np.ndarray:
        """Max confidence aggregation."""
        # Get class with maximum confidence for each segment
        max_class = np.argmax(predictions, axis=1)
        
        # Get the most frequent class
        most_frequent = np.bincount(max_class).argmax()
        
        # Get segments belonging to the most frequent class
        mask = (max_class == most_frequent)
        filtered_preds = predictions[mask]
        
        # If no segments match, fall back to mean aggregation
        if len(filtered_preds) == 0:
            return self._mean_aggregation(predictions)
        
        # Return prediction with maximum confidence for the most frequent class
        max_conf_idx = np.argmax(filtered_preds[:, most_frequent])
        return filtered_preds[max_conf_idx]
    
    def _voting_aggregation(self, predictions: np.ndarray, 
                           segment_info: Optional[Dict] = None) -> np.ndarray:
        """Hard voting aggregation."""
        # Get class with maximum confidence for each segment
        votes = np.argmax(predictions, axis=1)
        
        # Count votes for each class
        vote_counts = np.bincount(votes, minlength=predictions.shape[1])
        
        # Create zeroed prediction
        result = np.zeros(predictions.shape[1])
        
        # Set probability 1.0 for the winning class
        result[np.argmax(vote_counts)] = 1.0
        
        return result
    
    def _weighted_aggregation(self, predictions: np.ndarray, 
                             segment_info: Dict) -> np.ndarray:
        """
        Weighted aggregation based on segment quality or confidence.
        
        Args:
            predictions: Array of predictions
            segment_info: Dictionary with weights for each segment
        """
        if segment_info is None or 'weights' not in segment_info:
            return self._mean_aggregation(predictions)
        
        weights = np.array(segment_info['weights'])
        if len(weights) != len(predictions):
            return self._mean_aggregation(predictions)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Apply weights
        weighted_preds = predictions * weights[:, np.newaxis]
        
        return np.sum(weighted_preds, axis=0)


class VideoClassifier:
    """Class for classifying videos as deepfake or real."""
    
    def __init__(self, model, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the video classifier.
        
        Args:
            model: TensorFlow model for classification
            config_manager: Configuration manager instance
        """
        self.model = model
        self.config = config_manager or get_config()
        
        # Default parameters
        self.batch_size = self.config.get_model_params().get('batch_size', 32)
        self.frame_count = self.config.get_preprocessing_params().get('frames_per_clip', 20)
        
        # Prediction aggregator
        self.aggregator = PredictionAggregator(strategy='mean')
    
    def classify_folder(self, folder_path: str, 
                      output_path: Optional[str] = None) -> ClassificationResults:
        """
        Classify all videos in a folder.
        
        Args:
            folder_path: Path to the folder with videos
            output_path: Path to save results (optional)
            
        Returns:
            ClassificationResults object
        """
        # Get all video files
        filenames = sorted([f for f in os.listdir(folder_path) 
                          if f.endswith(('.mp4', '.avi'))])
        
        if not filenames:
            logger.warning(f"No video files found in {folder_path}")
            return ClassificationResults({})
        
        logger.info(f"Classifying {len(filenames)} videos from {folder_path}")
        
        # Load videos
        videos = DataUtilities.retrieve_data(folder_path)
        
        # Split into frames
        clips = DataUtilities.split_frames(videos, self.frame_count)
        
        if not clips:
            logger.warning(f"No valid clips found in {folder_path}")
            return ClassificationResults({})
        
        # Classify videos
        predictions = self._classify_clips(clips, filenames)
        
        # Create results object
        results = ClassificationResults(predictions)
        
        # Print results
        results.print_results()
        
        # Save results if output path is provided
        if output_path:
            results.save_to_csv(output_path)
        
        return results
    
    def _classify_clips(self, clips: List[np.ndarray], 
                      filenames: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Classify video clips.
        
        Args:
            clips: List of video clips
            filenames: List of filenames
            
        Returns:
            Dictionary of predictions
        """
        # Make sure clips is a numpy array
        clips_array = np.array(clips)
        
        # Get predictions
        try:
            # Use model's predict method with batching
            raw_predictions = self.model.predict(clips_array, batch_size=self.batch_size)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {}
        
        # Average predictions per video (assuming 6 clips per video)
        n_averaged_elements = 6
        aggregated_predictions = {}
        
        for i in range(0, len(raw_predictions), n_averaged_elements):
            # Get slice of predictions
            end_idx = min(i + n_averaged_elements, len(raw_predictions))
            pred_slice = raw_predictions[i:end_idx]
            
            # Skip if slice is empty
            if len(pred_slice) == 0:
                continue
            
            # Calculate video index
            video_idx = i // n_averaged_elements
            
            # Skip if video index is out of bounds
            if video_idx >= len(filenames):
                continue
            
            # Use aggregator to get final prediction
            aggregated = self.aggregator.aggregate(pred_slice)
            
            # Store prediction
            filename = filenames[video_idx]
            aggregated_predictions[filename] = {
                'Real': float(aggregated[0]),
                'Deepfake': float(aggregated[1])
            }
        
        return aggregated_predictions
    
    def classify_video(self, video_path: str) -> Dict[str, float]:
        """
        Classify a single video.
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary with classification probabilities
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {'Real': 0.0, 'Deepfake': 0.0}
        
        # Load video
        try:
            video = VideoUtilities.init_video(video_path)
            frames = []
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)
            
            video.release()
            
            if len(frames) == 0:
                logger.error(f"No frames found in video: {video_path}")
                return {'Real': 0.0, 'Deepfake': 0.0}
            
            # Convert to numpy array
            video_array = np.array(frames, dtype=np.float32)
            
            # Split into clips
            clips = DataUtilities.split_frames([video_array], self.frame_count)
            
            if len(clips) == 0:
                logger.error(f"No valid clips found in video: {video_path}")
                return {'Real': 0.0, 'Deepfake': 0.0}
            
            # Make predictions
            predictions = self.model.predict(np.array(clips), batch_size=self.batch_size)
            
            # Aggregate predictions
            aggregated = self.aggregator.aggregate(predictions)
            
            return {
                'Real': float(aggregated[0]),
                'Deepfake': float(aggregated[1])
            }
            
        except Exception as e:
            logger.error(f"Error classifying video {video_path}: {e}")
            return {'Real': 0.0, 'Deepfake': 0.0}
    
    def set_aggregation_strategy(self, strategy: str) -> None:
        """
        Set the aggregation strategy.
        
        Args:
            strategy: Aggregation strategy
        """
        self.aggregator = PredictionAggregator(strategy=strategy)
    
    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the batch size for prediction.
        
        Args:
            batch_size: Batch size
        """
        self.batch_size = batch_size
    
    def set_frame_count(self, frame_count: int) -> None:
        """
        Set the number of frames per clip.
        
        Args:
            frame_count: Number of frames per clip
        """
        self.frame_count = frame_count
    
    def set_model(self, model) -> None:
        """
        Set the classification model.
        
        Args:
            model: TensorFlow model
        """
        self.model = model


class EnsembleClassifier:
    """Class for ensemble classification using multiple models."""
    
    def __init__(self, models: List, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the ensemble classifier.
        
        Args:
            models: List of TensorFlow models
            config_manager: Configuration manager instance
        """
        self.models = models
        self.config = config_manager or get_config()
        self.classifiers = [VideoClassifier(model, self.config) for model in models]
        
        # Default to mean aggregation for ensemble
        self.ensemble_strategy = 'mean'
        self.aggregator = PredictionAggregator(strategy=self.ensemble_strategy)
    
    def classify_folder(self, folder_path: str, 
                      output_path: Optional[str] = None) -> ClassificationResults:
        """
        Classify all videos in a folder using ensemble of models.
        
        Args:
            folder_path: Path to the folder with videos
            output_path: Path to save results (optional)
            
        Returns:
            ClassificationResults object
        """
        if not self.models:
            logger.error("No models in ensemble")
            return ClassificationResults({})
        
        # Get all video files
        filenames = sorted([f for f in os.listdir(folder_path) 
                          if f.endswith(('.mp4', '.avi'))])
        
        if not filenames:
            logger.warning(f"No video files found in {folder_path}")
            return ClassificationResults({})
        
        logger.info(f"Ensemble classifying {len(filenames)} videos from {folder_path}")
        
        # Collect predictions from all models
        all_predictions = {}
        
        for i, classifier in enumerate(self.classifiers):
            logger.info(f"Running model {i+1}/{len(self.classifiers)}")
            model_results = classifier.classify_folder(folder_path)
            
            # Store predictions for each video
            for filename, probs in model_results.predictions.items():
                if filename not in all_predictions:
                    all_predictions[filename] = []
                
                # Add predictions to the list
                all_predictions[filename].append([probs['Real'], probs['Deepfake']])
        
        # Aggregate predictions from all models
        ensemble_predictions = {}
        
        for filename, preds in all_predictions.items():
            # Convert to numpy array
            preds_array = np.array(preds)
            
            # Aggregate predictions
            aggregated = self.aggregator.aggregate(preds_array)
            
            # Store prediction
            ensemble_predictions[filename] = {
                'Real': float(aggregated[0]),
                'Deepfake': float(aggregated[1])
            }
        
        # Create results object
        results = ClassificationResults(ensemble_predictions)
        
        # Print results
        results.print_results()
        
        # Save results if output path is provided
        if output_path:
            results.save_to_csv(output_path)
        
        return results
    
    def set_ensemble_strategy(self, strategy: str) -> None:
        """
        Set the ensemble aggregation strategy.
        
        Args:
            strategy: Aggregation strategy
        """
        self.ensemble_strategy = strategy
        self.aggregator = PredictionAggregator(strategy=strategy)
    
    def add_model(self, model) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: TensorFlow model
        """
        self.models.append(model)
        self.classifiers.append(VideoClassifier(model, self.config))
    
    def remove_model(self, index: int) -> None:
        """
        Remove a model from the ensemble.
        
        Args:
            index: Index of the model to remove
        """
        if 0 <= index < len(self.models):
            self.models.pop(index)
            self.classifiers.pop(index)


# Legacy Classifier class for backwards compatibility
class Classifier:
    """Legacy Classifier class for backwards compatibility."""
    
    def __init__(self, model, folder, frames):
        """
        Initialize the legacy classifier.
        
        Args:
            model: Model for classification
            folder: Folder with videos
            frames: Number of frames per clip
        """
        self.model = model
        self.folder = folder
        self.frames = frames
        self.filenames = sorted(os.listdir(folder))
        
        # Create modern classifier
        self.classifier = VideoClassifier(model)
        self.classifier.set_frame_count(frames)
        
        # Load videos
        self.unknown_videos = DataUtilities.retrieve_data(folder)
        self.unknown_clips = DataUtilities.split_frames(self.unknown_videos, frames)
    
    def classify_videos(self) -> Dict[str, Dict[str, float]]:
        """
        Classify videos in the folder.
        
        Returns:
            Dictionary of predictions
        """
        try:
            return self.classifier._classify_clips(self.unknown_clips, self.filenames)
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            print(f"Error: {e}")
            return {}
    
    def set_frames(self, frames: int) -> None:
        """
        Set the number of frames per clip.
        
        Args:
            frames: Number of frames per clip
        """
        self.frames = frames
        self.unknown_clips = DataUtilities.split_frames(self.unknown_videos, frames)
        self.classifier.set_frame_count(frames)
    
    def set_folder(self, folder: str) -> None:
        """
        Set the folder with videos.
        
        Args:
            folder: Folder path
        """
        self.folder = folder
        self.filenames = sorted(os.listdir(folder))
        self.unknown_videos = DataUtilities.retrieve_data(folder)
        self.unknown_clips = DataUtilities.split_frames(self.unknown_videos, self.frames)
    
    def set_model(self, model) -> None:
        """
        Set the classification model.
        
        Args:
            model: TensorFlow model
        """
        self.model = model
        self.classifier.set_model(model)
    
    def get_frames(self) -> int:
        """
        Get the number of frames per clip.
        
        Returns:
            Number of frames per clip
        """
        return self.frames
    
    def get_folder(self) -> str:
        """
        Get the folder path.
        
        Returns:
            Folder path
        """
        return self.folder
    
    def get_model(self):
        """
        Get the classification model.
        
        Returns:
            TensorFlow model
        """
        return self.model


# If used directly
if __name__ == "__main__":
    import argparse
    import tensorflow as tf
    
    parser = argparse.ArgumentParser(description='Thirdeye classification module')
    parser.add_argument('--folder', type=str, required=True,
                       help='Folder with videos to classify')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the model file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save classification results')
    parser.add_argument('--strategy', type=str, default='mean',
                       choices=['mean', 'median', 'max', 'voting', 'weighted'],
                       help='Aggregation strategy')
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder):
        print(f"Folder not found: {args.folder}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Load model
    try:
        model = tf.keras.models.load_model(args.model)
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create classifier
    classifier = VideoClassifier(model)
    classifier.set_aggregation_strategy(args.strategy)
    
    # Classify videos
    results = classifier.classify_folder(args.folder, args.output)
    
    # Plot results
    if args.output:
        output_dir = os.path.dirname(args.output)
        results.plot_confidence_histogram(os.path.join(output_dir, 'confidence_histogram.png'))
        results.plot_top_predictions(n=5, output_path=os.path.join(output_dir, 'top_predictions.png'))
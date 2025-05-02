import os
import sys
import logging
import numpy as np
import pandas as pd
import json
import time
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import our updated modules
from config_manager import ConfigManager, get_config
from preprocessing import Preprocessor
from networks import DeepfakeDetectionNetwork, NetworkFactory
from classify import VideoClassifier, EnsembleClassifier, ClassificationResults
from evaluate import Evaluator
from utilities import DataUtilities, FileSystemUtilities, VideoUtilities

"""
Main module for Thirdeye deepfake detection system.
Provides a unified interface for all system components.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thirdeye.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Thirdeye")


class Thirdeye:
    """Main class for the Thirdeye deepfake detection system."""
    
    def __init__(self, pre_p: bool = False, force_t: bool = False, 
                network: str = 'odin_v1', evaluate: bool = False, 
                max_for_class: int = 100000, frame_clip: int = 3,
                config_path: str = 'config.yaml'):
        """
        Initialize the Thirdeye system.
        
        Args:
            pre_p: Whether to perform preprocessing
            force_t: Whether to force training of the model
            network: Name of the network architecture to use
            evaluate: Whether to evaluate the model
            max_for_class: Maximum number of samples per class for training
            frame_clip: Number of frames per clip
            config_path: Path to the configuration file
        """
        self.PRE_PROCESSING = pre_p
        self.FORCE_TRAIN = force_t
        self.EVALUATE = evaluate
        self.network_name = network
        self.title = network.capitalize()
        self.MAX_FOR_CLASS = max_for_class
        self.FRAME_CLIP = frame_clip
        
        # Initialize configuration
        self.config = get_config(config_path)
        
        # Initialize components
        self.preprocessor = Preprocessor(config_path)
        self.network = DeepfakeDetectionNetwork(config_path, architecture=network)
        self.model = None
        
        # Initialize additional components
        self.classifier = None
        self.evaluator = None
        
        # Initialize data paths
        self._initialize_paths()
        
        # Process initialization options
        if self.PRE_PROCESSING:
            self.perform_preprocessing()
        
        if self.FORCE_TRAIN:
            self.train()
        else:
            self.load()
        
        if self.EVALUATE and (self.model is not None):
            self.evaluate()
    
    def _initialize_paths(self) -> None:
        """Initialize paths and directories."""
        # Create necessary directories
        paths = self.config.get_all_paths()
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
    
    def perform_preprocessing(self) -> None:
        """Perform preprocessing on training and testing data."""
        try:
            logger.info("Preprocessing training data")
            success = self.preprocessor.preprocess(1)
            
            logger.info("Preprocessing testing data")
            success = self.preprocessor.preprocess(2)
            
            if success:
                logger.info("Preprocessing completed successfully")
            else:
                logger.error("Preprocessing failed")
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            print(f"Error during preprocessing: {e}")
    
    def train(self) -> None:
        """Train the active model."""
        try:
            logger.info(f"Training {self.title}")
            
            # Prepare input data
            train_x, eval_x, train_y, eval_y = self._prepare_input_data(self.MAX_FOR_CLASS, self.FRAME_CLIP)
            
            if len(train_x) == 0 or len(train_y) == 0:
                logger.error("No training data available")
                print("No training data available")
                return
            
            # Build model
            input_shape = train_x[0].shape
            self.network.build_model(input_shape)
            self.model = self.network.model
            
            # Train model
            self.network.train(train_x, train_y, eval_x, eval_y)
            
            # Evaluate model
            if self.EVALUATE:
                self.evaluate(eval_x, eval_y, show=False)
            
            logger.info(f"Training of {self.title} completed")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            print(f"Error during training: {e}")
    
    def _prepare_input_data(self, max_samples: int = 100000, 
                         frame_clip: int = 3, test: bool = False, 
                         flip: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare input data for training or testing.
        
        Args:
            max_samples: Maximum number of samples per class
            frame_clip: Number of frames per clip
            test: Whether to prepare test data
            flip: Whether to apply flip augmentation
            
        Returns:
            Tuple of (train_x, eval_x, train_y, eval_y) or (test_x, test_y) if test=True
        """
        # Enable memory growth for GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"Memory growth setting failed: {e}")
        
        paths = self.config.get_all_paths()
        
        if test:
            # Load test data
            logger.info("Loading test data")
            
            # Load deepfake test data
            df_data = DataUtilities.retrieve_data(paths['test_deepfake_faces'])
            
            if len(df_data) == 0:
                logger.warning("Test deepfake data folder is empty, using training split for validation")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Split into frames
            if frame_clip != -1:
                df_data = DataUtilities.split_frames(df_data, frame_clip)
            
            df_labels = np.array([[0, 1]] * len(df_data))  # One-hot encoding: [0, 1] for deepfake
            logger.info(f"Found {len(df_data)} test deepfake samples")
            
            # Load real test data
            real_data = DataUtilities.retrieve_data(paths['test_real_faces'])
            
            # Split into frames
            if frame_clip != -1:
                real_data = DataUtilities.split_frames(real_data, frame_clip)
            
            real_labels = np.array([[1, 0]] * len(real_data))  # One-hot encoding: [1, 0] for real
            logger.info(f"Found {len(real_data)} test real samples")
            
            # Combine and shuffle data
            test_x = np.array(df_data + real_data)
            test_y = np.vstack((df_labels, real_labels))
            
            # Shuffle data
            indices = np.random.permutation(len(test_x))
            test_x = test_x[indices]
            test_y = test_y[indices]
            
            return test_x, np.array([]), test_y, np.array([])
        
        else:
            # Load training data
            logger.info("Loading training data")
            
            # Load deepfake training data
            df_data = DataUtilities.retrieve_data(paths['train_deepfake_faces'])
            
            if len(df_data) == 0:
                logger.error("Training deepfake data folder is empty")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Load real training data
            real_data = DataUtilities.retrieve_data(paths['train_real_faces'])
            
            if len(real_data) == 0:
                logger.error("Training real data folder is empty")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Split into training and validation sets
            df_train, df_val, _, _ = train_test_split(
                df_data, np.arange(len(df_data)), test_size=0.1, random_state=42
            )
            
            real_train, real_val, _, _ = train_test_split(
                real_data, np.arange(len(real_data)), test_size=0.1, random_state=42
            )
            
            # Split into frames
            if frame_clip != -1:
                df_train = DataUtilities.split_frames(df_train, frame_clip)
                real_train = DataUtilities.split_frames(real_train, frame_clip)
                df_val = DataUtilities.split_frames(df_val, frame_clip)
                real_val = DataUtilities.split_frames(real_val, frame_clip)
            
            # Apply flip augmentation if requested
            if flip:
                logger.info("Applying flip augmentation")
                df_train_flipped = self.flip_duplicate(df_train)
                real_train_flipped = self.flip_duplicate(real_train)
                
                df_train = df_train + df_train_flipped
                real_train = real_train + real_train_flipped
            
            logger.info(f"Found {len(df_train)} training deepfake samples")
            logger.info(f"Found {len(real_train)} training real samples")
            logger.info(f"Found {len(df_val)} validation deepfake samples")
            logger.info(f"Found {len(real_val)} validation real samples")
            
            # Limit the number of samples per class
            df_train = df_train[:max_samples]
            real_train = real_train[:max_samples]
            
            # Create labels (one-hot encoded)
            df_train_labels = np.array([[0, 1]] * len(df_train))  # [0, 1] for deepfake
            real_train_labels = np.array([[1, 0]] * len(real_train))  # [1, 0] for real
            
            df_val_labels = np.array([[0, 1]] * len(df_val))
            real_val_labels = np.array([[1, 0]] * len(real_val))
            
            # Combine training data
            train_x = np.array(df_train + real_train)
            train_y = np.vstack((df_train_labels, real_train_labels))
            
            # Combine validation data
            eval_x = np.array(df_val + real_val)
            eval_y = np.vstack((df_val_labels, real_val_labels))
            
            # Shuffle data
            train_indices = np.random.permutation(len(train_x))
            train_x = train_x[train_indices]
            train_y = train_y[train_indices]
            
            eval_indices = np.random.permutation(len(eval_x))
            eval_x = eval_x[eval_indices]
            eval_y = eval_y[eval_indices]
            
            return train_x, eval_x, train_y, eval_y
    
    def load(self) -> None:
        """Load the saved model."""
        try:
            # Try to load the model
            self.network.load_model()
            self.model = self.network.model
            
            if self.model is not None:
                logger.info(f"Model {self.title} loaded successfully")
            else:
                logger.warning(f"No saved model found for {self.title}, training is required")
                if self.FORCE_TRAIN:
                    logger.info("Force training is enabled, starting training")
                    self.train()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")
            
            if self.FORCE_TRAIN:
                logger.info("Force training is enabled, starting training")
                self.train()
    
    def evaluate(self, eval_x: Optional[np.ndarray] = None, 
                eval_y: Optional[np.ndarray] = None, 
                show: bool = True) -> None:
        """
        Evaluate the model.
        
        Args:
            eval_x: Evaluation data (optional)
            eval_y: Evaluation labels (optional)
            show: Whether to show evaluation plots
        """
        try:
            if self.model is None:
                logger.error("No model available for evaluation")
                print("No model available for evaluation")
                return
            
            # Create evaluator if needed
            if self.evaluator is None:
                self.evaluator = Evaluator(self.model, self.config, show_plots=show)
            else:
                self.evaluator.set_model(self.model)
                self.evaluator.set_show_plots(show)
            
            # Prepare evaluation data if not provided
            if eval_x is None or eval_y is None:
                eval_x, _, eval_y, _ = self._prepare_input_data(
                    self.MAX_FOR_CLASS, self.FRAME_CLIP, test=True
                )
            
            if len(eval_x) == 0 or len(eval_y) == 0:
                logger.error("No evaluation data available")
                print("No evaluation data available")
                return
            
            # Load training history
            history_path = os.path.join(
                self.config.config['data_paths']['models_dir'],
                f"{self.network_name}_history.npz"
            )
            
            # Plot training history if available
            if os.path.exists(history_path):
                history = np.load(history_path)
                history_dict = {key: history[key] for key in history.files}
                self.evaluator.plot_training_history(history_dict, self.title)
            
            # Evaluate model
            self.evaluator.evaluate_model(eval_x, eval_y, self.title)
            
            logger.info(f"Evaluation of {self.title} completed")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            print(f"Error during evaluation: {e}")
    
    def classify(self, output_dir: Optional[str] = None) -> Optional[ClassificationResults]:
        """
        Classify unknown videos.
        
        Args:
            output_dir: Directory to save classification results (optional)
            
        Returns:
            ClassificationResults object if successful, None otherwise
        """
        try:
            # Check if unknown data needs preprocessing
            unknown_paths = self.config.get_paths('unknown')
            
            if len(os.listdir(unknown_paths['raw'])) > 0:
                logger.info("Preprocessing unknown videos")
                self.preprocessor.preprocess(3)
            
            # Check if faces folder is empty
            if len(os.listdir(unknown_paths['faces'])) == 0:
                logger.error("No unknown videos to classify")
                print("No unknown videos to classify")
                return None
            
            # Create classifier if needed
            if self.classifier is None:
                self.classifier = VideoClassifier(self.model, self.config)
            else:
                self.classifier.set_model(self.model)
            
            # Set frame count
            self.classifier.set_frame_count(self.FRAME_CLIP)
            
            # Classify videos
            results = self.classifier.classify_folder(unknown_paths['faces'], output_dir)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            print(f"Error during classification: {e}")
            return None
    
    def flip_duplicate(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply flip augmentation to data.
        
        Args:
            data: List of videos
            
        Returns:
            List of flipped videos
        """
        flipped_videos = []
        
        for video in data:
            new_video = []
            for frame in video:
                new_frame = cv2.flip(frame, 1)  # Horizontal flip
                new_video.append(new_frame)
            flipped_videos.append(np.array(new_video))
        
        return flipped_videos
    
    def set_network(self, network: str) -> None:
        """
        Switch to a different network architecture.
        
        Args:
            network: Name of the network architecture
        """
        self.network_name = network
        self.title = network.capitalize()
        
        try:
            # Create new network
            self.network = DeepfakeDetectionNetwork(
                self.config.config_path, architecture=network
            )
            
            if self.FORCE_TRAIN:
                logger.info(f"Force train is enabled, training new network {self.title}")
                self.train()
            else:
                self.load()
                
        except Exception as e:
            logger.error(f"Error setting network: {e}")
            print(f"Error setting network: {e}")
    
    def set_frame_clip(self, frame_clip: int) -> None:
        """
        Set the number of frames per clip.
        
        Args:
            frame_clip: Number of frames per clip
        """
        self.FRAME_CLIP = frame_clip
        
        if self.classifier is not None:
            self.classifier.set_frame_count(frame_clip)
    
    def set_max_for_class(self, max_for_class: int) -> None:
        """
        Set the maximum number of samples per class for training.
        
        Args:
            max_for_class: Maximum number of samples per class
        """
        self.MAX_FOR_CLASS = max_for_class
    
    def get_max_for_class(self) -> int:
        """
        Get the maximum number of samples per class for training.
        
        Returns:
            Maximum number of samples per class
        """
        return self.MAX_FOR_CLASS
    
    def get_frame_clip(self) -> int:
        """
        Get the number of frames per clip.
        
        Returns:
            Number of frames per clip
        """
        return self.FRAME_CLIP
    
    def get_network(self) -> Dict[str, tf.keras.Model]:
        """
        Get the current network.
        
        Returns:
            Dictionary with network name and model
        """
        return {self.title: self.model}
    
    def create_ensemble(self, network_names: List[str]) -> Optional[EnsembleClassifier]:
        """
        Create an ensemble classifier with multiple models.
        
        Args:
            network_names: List of network names to include in the ensemble
            
        Returns:
            EnsembleClassifier object if successful, None otherwise
        """
        try:
            models = []
            
            # Load each model
            for name in network_names:
                network = DeepfakeDetectionNetwork(
                    self.config.config_path, architecture=name
                )
                network.load_model()
                
                if network.model is not None:
                    models.append(network.model)
                else:
                    logger.warning(f"Model {name} could not be loaded")
            
            if not models:
                logger.error("No models loaded for ensemble")
                return None
            
            # Create ensemble classifier
            ensemble = EnsembleClassifier(models, self.config)
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            print(f"Error creating ensemble: {e}")
            return None
    
    def compare_networks(self, network_names: List[str]) -> None:
        """
        Compare multiple network architectures on the test data.
        
        Args:
            network_names: List of network names to compare
        """
        try:
            # Prepare test data
            test_x, _, test_y, _ = self._prepare_input_data(
                self.MAX_FOR_CLASS, self.FRAME_CLIP, test=True
            )
            
            if len(test_x) == 0 or len(test_y) == 0:
                logger.error("No test data available for comparison")
                print("No test data available for comparison")
                return
            
            # Load models
            models = {}
            
            for name in network_names:
                network = DeepfakeDetectionNetwork(
                    self.config.config_path, architecture=name
                )
                network.load_model()
                
                if network.model is not None:
                    models[name] = network.model
                else:
                    logger.warning(f"Model {name} could not be loaded")
            
            if not models:
                logger.error("No models loaded for comparison")
                return
            
            # Create evaluator
            evaluator = Evaluator(None, self.config)
            
            # Compare models
            evaluator.compare_models(models, test_x, test_y)
            
            logger.info("Model comparison completed")
            
        except Exception as e:
            logger.error(f"Error comparing networks: {e}")
            print(f"Error comparing networks: {e}")
    
    def process_video(self, video_path: str) -> Dict[str, float]:
        """
        Process a single video through the entire pipeline.
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary with classification probabilities
        """
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return {'Real': 0.0, 'Deepfake': 0.0}
            
            # Create temporary directory
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
            # Process video
            face_path = self.preprocessor.process_single_video(video_path, temp_dir)
            
            if face_path is None:
                logger.error(f"Failed to process video: {video_path}")
                return {'Real': 0.0, 'Deepfake': 0.0}
            
            # Create classifier if needed
            if self.classifier is None:
                self.classifier = VideoClassifier(self.model, self.config)
            else:
                self.classifier.set_model(self.model)
            
            # Classify video
            result = self.classifier.classify_video(face_path)
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            print(f"Error processing video: {e}")
            return {'Real': 0.0, 'Deepfake': 0.0}
    
    # Legacy methods for backward compatibility
    def prepare_rgb_input(self, total_data: int = 1000, frame_clip: int = -1, 
                       test: bool = False, flip: bool = False) -> Union[
                           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                           Tuple[np.ndarray, np.ndarray]
                       ]:
        """
        Legacy method for backward compatibility.
        
        Args:
            total_data: Maximum number of samples per class
            frame_clip: Number of frames per clip
            test: Whether to prepare test data
            flip: Whether to apply flip augmentation
            
        Returns:
            Data tuple in the format expected by the original code
        """
        train_x, eval_x, train_y, eval_y = self._prepare_input_data(
            total_data, frame_clip, test, flip
        )
        
        if test:
            return train_x, train_y
        else:
            return train_x, eval_x, train_y, eval_y
    
    def prepare_mv_input(self, total_data: int = 1000, frame_clip: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy method for motion vector input preparation.
        
        Args:
            total_data: Maximum number of samples per class
            frame_clip: Number of frames per clip
            
        Returns:
            Tuple of (data, labels)
        """
        # This method is no longer used but kept for backward compatibility
        logger.warning("prepare_mv_input is deprecated and may not work as expected")
        
        paths = self.config.get_all_paths()
        
        # Load deepfake motion vectors
        df_data = DataUtilities.retrieve_data(paths['train_deepfake_mv'], rgb=False)
        
        # Split into frames
        if frame_clip != -1:
            df_data = DataUtilities.split_frames(df_data, frame_clip)
        
        df_labels = np.array([[0, 1]] * len(df_data))
        logger.info(f"Found {len(df_data)} deepfake motion vectors")
        
        # Load real motion vectors
        real_data = DataUtilities.retrieve_data(paths['train_real_mv'], rgb=False)
        
        # Split into frames
        if frame_clip != -1:
            real_data = DataUtilities.split_frames(real_data, frame_clip)
        
        real_labels = np.array([[1, 0]] * len(real_data))
        logger.info(f"Found {len(real_data)} real motion vectors")
        
        # Limit data
        df_data = df_data[:total_data]
        df_labels = df_labels[:total_data]
        real_data = real_data[:total_data]
        real_labels = real_labels[:total_data]
        
        # Combine data
        train_x = np.array(df_data + real_data)
        train_y = np.vstack((df_labels, real_labels))
        
        # Shuffle data
        indices = np.random.permutation(len(train_x))
        train_x = train_x[indices]
        train_y = train_y[indices]
        
        return train_x, train_y


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Thirdeye deepfake detection system')
    
    # Main operation modes
    parser.add_argument('--preprocess', action='store_true',
                       help='Perform preprocessing')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the model')
    parser.add_argument('--classify', action='store_true',
                       help='Classify unknown videos')
    
    # Model configuration
    parser.add_argument('--network', type=str, default='odin_v1',
                       help='Network architecture to use')
    parser.add_argument('--frame-clip', type=int, default=3,
                       help='Number of frames per clip')
    parser.add_argument('--max-samples', type=int, default=14600,
                       help='Maximum number of samples per class for training')
    
    # Advanced operations
    parser.add_argument('--ensemble', nargs='+',
                       help='Create an ensemble classifier with specified models')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple network architectures')
    parser.add_argument('--video', type=str,
                       help='Process a single video')
    
    # Output configuration
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create Thirdeye instance
    thirdeye = Thirdeye(
        pre_p=args.preprocess,
        force_t=args.train,
        network=args.network,
        evaluate=args.evaluate,
        max_for_class=args.max_samples,
        frame_clip=args.frame_clip,
        config_path=args.config
    )
    
    # Execute requested operations
    if args.classify:
        thirdeye.classify(args.output)
    
    if args.ensemble:
        ensemble = thirdeye.create_ensemble(args.ensemble)
        if ensemble:
            ensemble.classify_folder(
                thirdeye.config.get_paths('unknown')['faces'],
                os.path.join(args.output or '.',  'ensemble_results.csv') if args.output else None
            )
    
    if args.compare:
        thirdeye.compare_networks(args.compare)
    
    if args.video:
        result = thirdeye.process_video(args.video)
        print(f"Video: {args.video}")
        print(f"Real: {result['Real'] * 100:.2f}%, Deepfake: {result['Deepfake'] * 100:.2f}%")
        print(f"Classification: {'Real' if result['Real'] > result['Deepfake'] else 'Deepfake'}")


if __name__ == "__main__":
    # Version information
    print("Thirdeye Deepfake Detection System")
    print("Version 2.0.0")
    print("© 2025 Mahesha Kulatunga")
    print("")
    
    # Run main function
    main()
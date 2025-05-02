import cv2
import os
import numpy as np
import csv
import ast
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Set
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
from functools import wraps
import time

"""
Utilities module for Thirdeye deepfake detection system.
Contains reusable functions for video processing, data management, and visualization.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thirdeye_utilities.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Utilities")


def timing_decorator(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run")
        return result
    return wrapper


class VideoUtilities:
    """Helper class for video operations."""
    
    @staticmethod
    def init_video(filepath: str) -> cv2.VideoCapture:
        """
        Initialize a video capture object.
        
        Args:
            filepath: Path to the video file
            
        Returns:
            OpenCV VideoCapture object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If the video can't be opened
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        video = cv2.VideoCapture(filepath)
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video: {filepath}")
        
        return video
    
    @staticmethod
    def get_video_properties(video_path: str) -> Dict[str, Any]:
        """
        Get properties of a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video properties (width, height, fps, frame_count)
        """
        video = VideoUtilities.init_video(video_path)
        
        properties = {
            'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': video.get(cv2.CAP_PROP_FPS),
            'frame_count': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        video.release()
        return properties
    
    @staticmethod
    def get_frame_values(directory_path: str) -> Dict[str, int]:
        """
        Get frame counts for all videos in a directory.
        
        Args:
            directory_path: Path to directory containing videos
            
        Returns:
            Dictionary mapping filenames to frame counts
        """
        frame_counts = {}
        video_files = [f for f in os.listdir(directory_path) 
                     if f.endswith(('.mp4', '.avi'))]
        
        for filename in tqdm(video_files, desc="Counting frames"):
            try:
                video_path = os.path.join(directory_path, filename)
                properties = VideoUtilities.get_video_properties(video_path)
                frame_counts[filename] = properties['frame_count']
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
        
        return frame_counts
    
    @staticmethod
    def flip_img(img: np.ndarray) -> np.ndarray:
        """
        Flip an image horizontally.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Horizontally flipped image
        """
        return cv2.flip(img, 1)  # 1 = horizontal flip, 0 = vertical, -1 = both
    
    @staticmethod
    def apply_transformations(frames: List[np.ndarray], 
                             transformations: List[str] = None) -> List[np.ndarray]:
        """
        Apply a series of transformations to video frames.
        
        Args:
            frames: List of frames as numpy arrays
            transformations: List of transformation names to apply
                             (supported: 'flip_h', 'flip_v', 'rotate90', 'rotate180')
            
        Returns:
            Transformed frames
        """
        if transformations is None:
            return frames
        
        result = frames.copy()
        
        for transform in transformations:
            if transform == 'flip_h':
                result = [cv2.flip(frame, 1) for frame in result]
            elif transform == 'flip_v':
                result = [cv2.flip(frame, 0) for frame in result]
            elif transform == 'rotate90':
                result = [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in result]
            elif transform == 'rotate180':
                result = [cv2.rotate(frame, cv2.ROTATE_180) for frame in result]
            else:
                logger.warning(f"Unknown transformation: {transform}")
        
        return result


class DataUtilities:
    """Helper class for data operations."""
    
    @staticmethod
    @timing_decorator
    def retrieve_data(folder: str, rgb: bool = True, 
                     mv_type: str = 'mag') -> List[np.ndarray]:
        """
        Load video or motion vector data from a folder.
        
        Args:
            folder: Directory containing data files
            rgb: If True, load RGB video data; if False, load motion vectors
            mv_type: Type of motion vectors to load ('mag' or 'ang')
            
        Returns:
            List of numpy arrays containing the loaded data
        """
        data = []
        sorted_files = sorted(os.listdir(folder))
        total_files = len(sorted_files)
        
        if total_files == 0:
            logger.warning(f"No files found in {folder}")
            return data
        
        if rgb:
            logger.info(f"Loading {total_files} videos from {folder}")
            
            for filename in tqdm(sorted_files, desc="Loading videos"):
                if not filename.endswith(('.mp4', '.avi')):
                    continue
                    
                try:
                    file_path = os.path.join(folder, filename)
                    cap = VideoUtilities.init_video(file_path)
                    
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    
                    if frames:
                        data.append(np.array(frames, dtype=np.float32))
                    cap.release()
                except Exception as e:
                    logger.error(f"Error loading video {filename}: {e}")
        else:
            logger.info(f"Loading {total_files} motion vector files from {folder}")
            
            for filename in tqdm(sorted_files, desc="Loading motion vectors"):
                if not filename.endswith('.csv'):
                    continue
                    
                try:
                    file_path = os.path.join(folder, filename)
                    
                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        mvs = []
                        
                        for row in reader:
                            frame = []
                            for data in row:
                                try:
                                    # Safely evaluate string representation of list
                                    data_list = ast.literal_eval(data)
                                    frame.append(data_list)
                                except (SyntaxError, ValueError) as e:
                                    logger.warning(f"Error parsing data in {filename}: {e}")
                            
                            if frame:
                                mvs.append(np.array(frame, dtype=np.float32))
                    
                    if mvs:
                        data.append(np.array(mvs))
                except Exception as e:
                    logger.error(f"Error loading motion vectors from {filename}: {e}")
        
        logger.info(f"Successfully loaded {len(data)} data items")
        return data
    
    @staticmethod
    def split_frames(videos: List[np.ndarray], chunk_size: int) -> List[np.ndarray]:
        """
        Split videos into chunks of specified frame count.
        
        Args:
            videos: List of videos as numpy arrays
            chunk_size: Number of frames per chunk
            
        Returns:
            List of video chunks
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        chunks = []
        
        for video in videos:
            # Skip videos with fewer frames than chunk_size
            if len(video) < chunk_size:
                continue
                
            # Create chunks
            video_chunks = [
                video[i:i + chunk_size] 
                for i in range(0, len(video), chunk_size)
                if i + chunk_size <= len(video)  # Only include complete chunks
            ]
            
            chunks.extend(video_chunks)
        
        logger.info(f"Split {len(videos)} videos into {len(chunks)} chunks of {chunk_size} frames")
        return chunks
    
    @staticmethod
    def normalize_video_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize video data.
        
        Args:
            data: Video data as numpy array
            method: Normalization method ('minmax', 'standard', or 'per_frame')
            
        Returns:
            Normalized data
        """
        if method == 'minmax':
            # Global min-max normalization to [0, 1]
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val > min_val:
                return (data - min_val) / (max_val - min_val)
            return data
            
        elif method == 'standard':
            # Global standardization (zero mean, unit variance)
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return (data - mean) / std
            return data
            
        elif method == 'per_frame':
            # Normalize each frame independently
            result = np.zeros_like(data)
            for i, frame in enumerate(data):
                min_val = np.min(frame)
                max_val = np.max(frame)
                if max_val > min_val:
                    result[i] = (frame - min_val) / (max_val - min_val)
                else:
                    result[i] = frame
            return result
            
        else:
            logger.warning(f"Unknown normalization method: {method}, using minmax")
            return DataUtilities.normalize_video_data(data, 'minmax')


class FileSystemUtilities:
    """Helper class for file system operations."""
    
    @staticmethod
    def clear_folder(folder: str, pattern: str = None) -> int:
        """
        Delete files in a folder.
        
        Args:
            folder: Directory to clear
            pattern: Optional file pattern to match (e.g., '*.mp4')
            
        Returns:
            Number of files deleted
        """
        if not os.path.exists(folder):
            logger.warning(f"Folder {folder} does not exist")
            return 0
        
        count = 0
        for filename in os.listdir(folder):
            if pattern and not Path(filename).match(pattern):
                continue
                
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        
        logger.info(f"Cleared {count} files from {folder}")
        return count
    
    @staticmethod
    def ensure_directories(directories: List[str]) -> None:
        """
        Ensure directories exist, creating them if necessary.
        
        Args:
            directories: List of directories to create
        """
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
    
    @staticmethod
    def get_processed_files(log_path: str) -> Set[str]:
        """
        Get set of already processed files from a log.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Set of processed filenames
        """
        processed = set()
        
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            processed.add(row[0])
            except Exception as e:
                logger.error(f"Error reading processed files log {log_path}: {e}")
        
        return processed
    
    @staticmethod
    def log_processed_file(log_path: str, filename: str) -> bool:
        """
        Log a processed file.
        
        Args:
            log_path: Path to the log file
            filename: Name of the processed file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename])
            return True
        except Exception as e:
            logger.error(f"Error logging processed file {filename}: {e}")
            return False


class VisualizationUtilities:
    """Helper class for visualization."""
    
    @staticmethod
    def plot_video_frames(video: np.ndarray, max_frames: int = 9, 
                         figsize: Tuple[int, int] = (15, 10),
                         title: str = None) -> plt.Figure:
        """
        Plot frames from a video.
        
        Args:
            video: Video as numpy array
            max_frames: Maximum number of frames to plot
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        frames_to_plot = min(len(video), max_frames)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(frames_to_plot)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        
        if title:
            fig.suptitle(title)
        
        # Flatten axes array for easier indexing
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Plot frames
        for i in range(frames_to_plot):
            frame = video[i]
            
            # Convert to uint8 if normalized
            if frame.dtype != np.uint8:
                if np.max(frame) <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert BGR to RGB
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(frames_to_plot, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_motion_vectors(mv_data: np.ndarray, max_frames: int = 5,
                          figsize: Tuple[int, int] = (15, 10),
                          title: str = None) -> plt.Figure:
        """
        Visualize motion vectors.
        
        Args:
            mv_data: Motion vector data
            max_frames: Maximum number of frames to plot
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        frames_to_plot = min(len(mv_data), max_frames)
        
        fig, axes = plt.subplots(1, frames_to_plot, figsize=figsize)
        
        if title:
            fig.suptitle(title)
        
        # Handle case where only one frame is plotted
        axes = [axes] if frames_to_plot == 1 else axes
        
        # Plot frames
        for i in range(frames_to_plot):
            frame = mv_data[i]
            
            # Normalize for visualization
            normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply colormap for better visualization
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(colored)
            axes[i].set_title(f"MV Frame {i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


# If used directly
if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Thirdeye utilities module')
    parser.add_argument('--video', type=str, help='Path to a video file for testing')
    
    args = parser.parse_args()
    
    if args.video:
        if os.path.exists(args.video):
            # Display video properties
            props = VideoUtilities.get_video_properties(args.video)
            print(f"Video properties: {props}")
            
            # Load and display frames
            cap = VideoUtilities.init_video(args.video)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if frames:
                video_array = np.array(frames)
                fig = VisualizationUtilities.plot_video_frames(video_array, title="Sample Frames")
                plt.show()
        else:
            print(f"Video file not found: {args.video}")
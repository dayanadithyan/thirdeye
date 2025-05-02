import time
import numpy as np
import os
import subprocess
import logging
from pathlib import Path
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import face_recognition
from typing import List, Tuple, Optional, Dict, Any, Union
import json
import csv
import sys
import yaml
from tqdm import tqdm
import concurrent.futures

"""
Preprocessing module for the Thirdeye deepfake detection system.
Handles all video preprocessing tasks including face detection, cropping, and motion vector extraction.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thirdeye_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Preprocessing")

class ConfigManager:
    """Handle configuration settings for the preprocessing pipeline."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file, or create default if not found.
        
        Returns:
            Dict containing configuration settings
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            # Default configuration
            default_config = {
                'data_paths': {
                    'base_dir': './Data/',
                    'figures_dir': './Figures/',
                    'models_dir': './Saved_Models/',
                },
                'preprocessing': {
                    'face_box_bias': 20,
                    'face_box_size': 100,
                    'frames_per_clip': 20,
                    'fps': 20,
                    'clip_duration': 1,
                    'batch_size': 32,
                },
                'splits': {
                    'train': {
                        'real_raw': './Data/TRAIN/REAL_RAW/',
                        'deepfake_raw': './Data/TRAIN/DF_RAW/',
                        'real_fps': './Data/TRAIN/REAL_RAW/FPS/',
                        'deepfake_fps': './Data/TRAIN/DF_RAW/FPS/',
                        'real_clips': './Data/TRAIN/REAL_CLIPS/',
                        'deepfake_clips': './Data/TRAIN/DF_CLIPS/',
                        'real_faces': './Data/TRAIN/REAL_SAMPLES/',
                        'deepfake_faces': './Data/TRAIN/DF_SAMPLES/',
                        'real_mv': './Data/TRAIN/REAL_MV/',
                        'deepfake_mv': './Data/TRAIN/DF_MV/',
                    },
                    'test': {
                        'real_raw': './Data/TEST/REAL_RAW/',
                        'deepfake_raw': './Data/TEST/DF_RAW/',
                        'real_fps': './Data/TEST/REAL_RAW/FPS/',
                        'deepfake_fps': './Data/TEST/DF_RAW/FPS/',
                        'real_clips': './Data/TEST/REAL_CLIPS/',
                        'deepfake_clips': './Data/TEST/DF_CLIPS/',
                        'real_faces': './Data/TEST/REAL_SAMPLES/',
                        'deepfake_faces': './Data/TEST/DF_SAMPLES/',
                        'real_mv': './Data/TEST/REAL_MV/',
                        'deepfake_mv': './Data/TEST/DF_MV/',
                    },
                    'unknown': {
                        'raw': './Data/UNKNOWN/UNKNOWN_RAW/',
                        'fps': './Data/UNKNOWN/UNKNOWN_FPS/',
                        'clips': './Data/UNKNOWN/UNKNOWN_CLIPS/',
                        'faces': './Data/UNKNOWN/UNKNOWN_SAMPLES/',
                    }
                }
            }
            
            # Create directories if they don't exist
            for split in default_config['splits']:
                for _, path in default_config['splits'][split].items():
                    os.makedirs(path, exist_ok=True)
            
            # Write default config
            with open(self.config_path, 'w') as file:
                yaml.dump(default_config, file)
            
            return default_config
    
    def get_paths(self, split: str) -> Dict[str, str]:
        """
        Get paths for a specific data split.
        
        Args:
            split: The data split ('train', 'test', or 'unknown')
            
        Returns:
            Dictionary of paths for the specified split
        """
        return self.config['splits'].get(split, {})
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """
        Get preprocessing parameters.
        
        Returns:
            Dictionary of preprocessing parameters
        """
        return self.config['preprocessing']


class VideoProcessor:
    """Process videos for face detection and extraction."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the video processor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.params = config_manager.get_preprocessing_params()
    
    def standardize_fps(self, input_path: str, output_path: str, fps: Optional[int] = None) -> bool:
        """
        Standardize the FPS of a video.
        
        Args:
            input_path: Path to the input video
            output_path: Path to save the processed video
            fps: Target frames per second (uses config value if None)
            
        Returns:
            True if successful, False otherwise
        """
        if fps is None:
            fps = self.params['fps']
        
        try:
            command = f"ffmpeg -i {input_path} -r {fps} -y {output_path}"
            subprocess.run(command, shell=True, check=True, 
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error standardizing FPS for {input_path}: {e}")
            return False
    
    def get_largest_face_size(self, video_path: str) -> Tuple[int, int]:
        """
        Find the largest face dimensions in a video.
        
        Args:
            video_path: Path to the video
            
        Returns:
            Tuple of (width, height) of the largest face
        """
        largest_face_height = 0
        largest_face_width = 0
        
        video = cv2.VideoCapture(video_path)
        
        try:
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    height = bottom - top
                    width = right - left
                    
                    largest_face_height = max(largest_face_height, height)
                    largest_face_width = max(largest_face_width, width)
        finally:
            video.release()
        
        return largest_face_width, largest_face_height
    
    def split_video(self, video_path: str, output_dir: str, 
                   clip_duration: Optional[int] = None) -> List[str]:
        """
        Split a video into clips of specified duration.
        
        Args:
            video_path: Path to the video
            output_dir: Directory to save the clips
            clip_duration: Duration of each clip in seconds (uses config value if None)
            
        Returns:
            List of paths to the created clips
        """
        if clip_duration is None:
            clip_duration = self.params['clip_duration']
        
        clip_paths = []
        
        try:
            with VideoFileClip(video_path) as video:
                # Calculate number of clips
                clip_count = int(video.duration / clip_duration)
                
                for i in range(clip_count):
                    start_time = i * clip_duration
                    end_time = start_time + clip_duration
                    
                    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_clip{i}.mp4")
                    clip = video.subclip(start_time, end_time)
                    clip.write_videofile(output_path, audio=False, codec='libx264', 
                                        logger=None, verbose=False)
                    clip_paths.append(output_path)
        except Exception as e:
            logger.error(f"Error splitting video {video_path}: {e}")
        
        return clip_paths
    
    def extract_face(self, video_path: str, output_path: str, 
                    box_bias: Optional[int] = None, 
                    box_size: Optional[int] = None,
                    min_frames: Optional[int] = None) -> bool:
        """
        Extract faces from a video and save to a new video.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the processed video
            box_bias: Extra pixels around the face
            box_size: Final size of the face box
            min_frames: Minimum number of frames with faces required
            
        Returns:
            True if successful, False otherwise
        """
        if box_bias is None:
            box_bias = self.params['face_box_bias']
        if box_size is None:
            box_size = self.params['face_box_size']
        if min_frames is None:
            min_frames = self.params['frames_per_clip']
        
        # Get video properties
        input_video = cv2.VideoCapture(video_path)
        fps = input_video.get(cv2.CAP_PROP_FPS)
        frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Find largest face size first
        largest_face_width, largest_face_height = self.get_largest_face_size(video_path)
        if largest_face_width == 0 or largest_face_height == 0:
            logger.warning(f"No faces found in {video_path}")
            input_video.release()
            return False
        
        # Reset video capture for processing
        input_video = cv2.VideoCapture(video_path)
        
        # Prepare for output video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        processed_frames = []
        frame_number = 0
        
        # Process each frame
        while True:
            ret, frame = input_video.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Convert to RGB for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                # Extract the first face
                top, right, bottom, left = face_locations[0]
                
                # Adjust to ensure consistent size with largest face
                if (right - left) < largest_face_width:
                    right = left + largest_face_width
                
                if (bottom - top) < largest_face_height:
                    bottom = top + largest_face_height
                
                # Add bias and crop
                try:
                    cropped_frame = frame[
                        max(0, top - box_bias):min(frame.shape[0], bottom + box_bias),
                        max(0, left - box_bias):min(frame.shape[1], right + box_bias)
                    ]
                    
                    # Resize to standard size
                    resized_frame = cv2.resize(cropped_frame, (box_size, box_size), 
                                              interpolation=cv2.INTER_LINEAR)
                    processed_frames.append(resized_frame)
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_number} in {video_path}: {e}")
            else:
                logger.debug(f"No face detected in frame {frame_number} of {video_path}")
        
        input_video.release()
        
        # Check if we have enough frames with faces
        if len(processed_frames) >= min_frames:
            # Write to output video
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (box_size, box_size))
            for frame in processed_frames:
                output_video.write(frame)
            output_video.release()
            return True
        elif len(processed_frames) >= min_frames * 0.75:
            # If we have at least 75% of required frames, duplicate the first frame
            logger.info(f"Duplicating frames for {video_path} ({len(processed_frames)}/{min_frames})")
            processed_frames.extend([processed_frames[0]] * (min_frames - len(processed_frames)))
            
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (box_size, box_size))
            for frame in processed_frames:
                output_video.write(frame)
            output_video.release()
            return True
        else:
            logger.warning(f"Insufficient face frames in {video_path}: {len(processed_frames)}/{min_frames}")
            return False
    
    def extract_motion_vectors(self, video_path: str, output_path: str, 
                              min_frames: Optional[int] = None) -> bool:
        """
        Extract motion vectors from a video and save as CSV.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the CSV file
            min_frames: Minimum number of frames required
            
        Returns:
            True if successful, False otherwise
        """
        if min_frames is None:
            min_frames = self.params['frames_per_clip']
        
        try:
            input_video = cv2.VideoCapture(video_path)
            
            # Initialize for optical flow
            ret, frame1 = input_video.read()
            if not ret:
                logger.error(f"Could not read video {video_path}")
                return False
            
            prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255
            
            frame_list = []
            mag_vectors = []
            
            # Process each frame
            while True:
                ret, frame2 = input_video.read()
                if not ret:
                    break
                
                next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Convert to polar coordinates
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Visualize flow
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                frame_list.append(rgb)
                mag_vectors.append(mag)
                
                # Update for next iteration
                prev_frame = next_frame
            
            input_video.release()
            
            # Check if we have enough frames
            if len(frame_list) == (min_frames - 1):
                # Save magnitude vectors as CSV
                with open(output_path, 'w', newline='') as outfile:
                    writer = csv.writer(outfile, delimiter=',')
                    for frame_mag in mag_vectors:
                        writer.writerow(frame_mag.tolist())
                return True
            else:
                logger.warning(f"Insufficient frames for motion vectors in {video_path}: {len(frame_list)}/{min_frames-1}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting motion vectors from {video_path}: {e}")
            return False


class Preprocessor:
    """
    Main preprocessing class that orchestrates the entire preprocessing pipeline.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.video_processor = VideoProcessor(self.config_manager)
        
        # Create required directories
        for split in ['train', 'test', 'unknown']:
            paths = self.config_manager.get_paths(split)
            for path in paths.values():
                os.makedirs(path, exist_ok=True)
    
    def preprocess(self, split_type: int) -> bool:
        """
        Run the preprocessing pipeline.
        
        Args:
            split_type: 1 for training, 2 for testing, 3 for unknown data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if split_type == 1:
                logger.info("Preprocessing training data")
                self._process_training_data()
            elif split_type == 2:
                logger.info("Preprocessing testing data")
                self._process_testing_data()
            elif split_type == 3:
                logger.info("Preprocessing unknown data")
                self._process_unknown_data()
            else:
                logger.error(f"Invalid split type: {split_type}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return False
    
    def _process_videos(self, raw_path: str, fps_path: str, clips_path: str, 
                       faces_path: str, mv_path: str = None) -> None:
        """
        Process a set of videos through the entire pipeline.
        
        Args:
            raw_path: Path to raw videos
            fps_path: Path to store FPS-standardized videos
            clips_path: Path to store video clips
            faces_path: Path to store face-extracted videos
            mv_path: Path to store motion vectors (optional)
        """
        # Get processed files to avoid reprocessing
        processed_files_path = os.path.join(os.path.dirname(raw_path), 'processed_files.csv')
        processed_files = set()
        
        if os.path.exists(processed_files_path):
            with open(processed_files_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        processed_files.add(row[0])
        
        # Get new files to process
        raw_files = [f for f in os.listdir(raw_path) 
                    if f.endswith('.mp4') and f not in processed_files]
        
        if not raw_files:
            logger.info(f"No new files to process in {raw_path}")
            return
        
        logger.info(f"Processing {len(raw_files)} new videos from {raw_path}")
        
        # Process each file
        for filename in tqdm(raw_files, desc="Processing videos"):
            input_path = os.path.join(raw_path, filename)
            fps_output_path = os.path.join(fps_path, filename)
            
            # Standardize FPS
            if self.video_processor.standardize_fps(input_path, fps_output_path):
                # Split into clips
                clip_paths = self.video_processor.split_video(fps_output_path, clips_path)
                
                # Process each clip
                for clip_path in clip_paths:
                    # Extract faces
                    face_output_path = os.path.join(faces_path, os.path.basename(clip_path))
                    if self.video_processor.extract_face(clip_path, face_output_path):
                        # Extract motion vectors if requested
                        if mv_path:
                            mv_output_path = os.path.join(mv_path, f"{os.path.basename(clip_path)}.csv")
                            self.video_processor.extract_motion_vectors(face_output_path, mv_output_path)
                
                # Mark as processed
                with open(processed_files_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename])
            
            # Clean up FPS file to save space
            if os.path.exists(fps_output_path):
                os.remove(fps_output_path)
    
    def _process_training_data(self) -> None:
        """Process training data (both real and deepfake)."""
        logger.info("Processing training data - Deepfakes")
        train_paths = self.config_manager.get_paths('train')
        
        self._process_videos(
            train_paths['deepfake_raw'],
            train_paths['deepfake_fps'],
            train_paths['deepfake_clips'],
            train_paths['deepfake_faces'],
            train_paths['deepfake_mv']
        )
        
        logger.info("Processing training data - Real")
        self._process_videos(
            train_paths['real_raw'],
            train_paths['real_fps'],
            train_paths['real_clips'],
            train_paths['real_faces'],
            train_paths['real_mv']
        )
    
    def _process_testing_data(self) -> None:
        """Process testing data (both real and deepfake)."""
        logger.info("Processing testing data - Deepfakes")
        test_paths = self.config_manager.get_paths('test')
        
        self._process_videos(
            test_paths['deepfake_raw'],
            test_paths['deepfake_fps'],
            test_paths['deepfake_clips'],
            test_paths['deepfake_faces'],
            test_paths['deepfake_mv']
        )
        
        logger.info("Processing testing data - Real")
        self._process_videos(
            test_paths['real_raw'],
            test_paths['real_fps'],
            test_paths['real_clips'],
            test_paths['real_faces'],
            test_paths['real_mv']
        )
    
    def _process_unknown_data(self) -> None:
        """Process unknown data for classification."""
        logger.info("Processing unknown data for classification")
        unknown_paths = self.config_manager.get_paths('unknown')
        
        self._process_videos(
            unknown_paths['raw'],
            unknown_paths['fps'],
            unknown_paths['clips'],
            unknown_paths['faces']
        )
    
    def process_single_video(self, video_path: str, output_dir: str) -> str:
        """
        Process a single video through the entire pipeline.
        
        Args:
            video_path: Path to the input video
            output_dir: Directory to save processed outputs
            
        Returns:
            Path to the processed face video
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(video_path)
        fps_path = os.path.join(output_dir, f"fps_{filename}")
        
        # Standardize FPS
        if self.video_processor.standardize_fps(video_path, fps_path):
            # Split into clips
            clip_paths = self.video_processor.split_video(fps_path, output_dir)
            
            # Process first clip
            if clip_paths:
                face_output_path = os.path.join(output_dir, f"face_{filename}")
                if self.video_processor.extract_face(clip_paths[0], face_output_path):
                    # Clean up temporary files
                    os.remove(fps_path)
                    for clip_path in clip_paths:
                        os.remove(clip_path)
                    
                    return face_output_path
        
        logger.error(f"Failed to process video {video_path}")
        return None


# Main execution block for direct script usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Thirdeye preprocessing module')
    parser.add_argument('--split', type=int, choices=[1, 2, 3], default=1,
                       help='Data split to process: 1=training, 2=testing, 3=unknown')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    preprocessor = Preprocessor(args.config)
    success = preprocessor.preprocess(args.split)
    
    if success:
        logger.info(f"Preprocessing completed successfully for split {args.split}")
    else:
        logger.error(f"Preprocessing failed for split {args.split}")
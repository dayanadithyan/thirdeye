import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

"""
Configuration management for Thirdeye deepfake detection system.
Replaces the original constants.py with a more flexible YAML-based configuration system.
"""

logger = logging.getLogger("Config")

class ConfigManager:
    """Configuration manager for Thirdeye system."""

    DEFAULT_CONFIG = {
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
        'model': {
            'learning_rate': 0.001,
            'epochs': 20,
            'batch_size': 32,
            'validation_split': 0.2,
            'l2_regularization': 0.001,
            'dropout_rate': 0.4,
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
        },
        'networks': {
            'providence_v1': {
                'filters1': 8,
                'filters2': 16,
                'conv2': True,
                'conv4': True,
                'nodes_1': 2048,
                'nodes_2': 512,
                'leaky': False,
                'epochs': 10,
            },
            'providence_v2': {
                'filters1': 8,
                'filters2': 16,
                'conv2': True,
                'conv4': True,
                'nodes_1': 256,
                'nodes_2': 128,
                'leaky': False,
                'epochs': 10,
            },
            'odin_v1': {
                'filters1': 8,
                'filters2': 16,
                'conv2': True,
                'conv4': False,
                'nodes_1': 32,
                'nodes_2': 16,
                'leaky': False,
                'epochs': 10,
            },
            'odin_v2': {
                'filters1': 8,
                'filters2': 16,
                'conv2': True,
                'conv4': False,
                'nodes_1': 32,
                'nodes_2': 16,
                'leaky': True,
                'epochs': 25,
            },
            'horus': {
                'filters1': 16,
                'filters2': 16,
                'conv2': False,
                'conv4': False,
                'nodes_1': 32,
                'nodes_2': 16,
                'leaky': False,
                'epochs': 20,
            },
        }
    }

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
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
            try:
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    logger.info(f"Loaded configuration from {self.config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {e}")
                logger.info("Using default configuration")
                return self._create_default_config()
        else:
            logger.info(f"Configuration file {self.config_path} not found, creating default")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create and save default configuration.
        
        Returns:
            Default configuration dictionary
        """
        config = self.DEFAULT_CONFIG
        
        # Create directories if they don't exist
        for split in config['splits']:
            for _, path in config['splits'][split].items():
                os.makedirs(path, exist_ok=True)
        
        # Save default configuration
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            logger.info(f"Saved default configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving default configuration to {self.config_path}: {e}")
        
        return config
    
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
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model training parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.config['model']
    
    def get_network_params(self, network_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific network architecture.
        
        Args:
            network_name: Name of the network
            
        Returns:
            Dictionary of network parameters
        """
        return self.config['networks'].get(network_name, {})
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save the current configuration to file.
        
        Args:
            config: Optional configuration to save (uses current if None)
            
        Returns:
            True if successful, False otherwise
        """
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_path}: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update the configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v
        
        try:
            update_nested_dict(self.config, updates)
            return self.save_config()
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_all_paths(self) -> Dict[str, str]:
        """
        Get all paths from the configuration.
        
        Returns:
            Dictionary of all paths
        """
        paths = {}
        
        # Add base directories
        for key, value in self.config['data_paths'].items():
            paths[key] = value
        
        # Add split-specific paths
        for split, split_paths in self.config['splits'].items():
            for key, value in split_paths.items():
                paths[f"{split}_{key}"] = value
        
        return paths


def get_config(config_path: str = 'config.yaml') -> ConfigManager:
    """
    Get a configuration manager instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


# For backwards compatibility with original constants.py
def get_constants() -> Dict[str, str]:
    """
    Get constants in the original format for backwards compatibility.
    
    Returns:
        Dictionary of constants
    """
    config = get_config()
    paths = config.get_all_paths()
    
    # Map to original constant names
    constants = {
        'DATA': paths['base_dir'],
        'FIGURES': paths['figures_dir'],
        'SAVED_MODELS': paths['models_dir'],
        
        # Training folders
        'RAW_DEEPFAKES': paths['train_deepfake_raw'],
        'RAW_REAL': paths['train_real_raw'],
        'TRAIN_DEEPFAKES': paths['train_deepfake_clips'],
        'TRAIN_REAL': paths['train_real_clips'],
        'TRAIN_FPS_DEEPFAKES': paths['train_deepfake_fps'],
        'TRAIN_FPS_REAL': paths['train_real_fps'],
        'TRAIN_SEPARATED_DF_FACES': paths['train_deepfake_faces'],
        'TRAIN_SEPARATED_REAL_FACES': paths['train_real_faces'],
        'TRAIN_MV_DF_FACES': paths['train_deepfake_mv'],
        'TRAIN_MV_REAL_FACES': paths['train_real_mv'],
        
        # Testing folders
        'TEST_RAW_DEEPFAKES': paths['test_deepfake_raw'],
        'TEST_RAW_REAL': paths['test_real_raw'],
        'TEST_DEEPFAKES': paths['test_deepfake_clips'],
        'TEST_REAL': paths['test_real_clips'],
        'TEST_FPS_DEEPFAKES': paths['test_deepfake_fps'],
        'TEST_FPS_REAL': paths['test_real_fps'],
        'TEST_SEPARATED_DF_FACES': paths['test_deepfake_faces'],
        'TEST_SEPARATED_REAL_FACES': paths['test_real_faces'],
        'TEST_MV_DF_FACES': paths['test_deepfake_mv'],
        'TEST_MV_REAL_FACES': paths['test_real_mv'],
        
        # Unknown video folders
        'UNKNOWN_RAW': paths['unknown_raw'],
        'UNKNOWN_FPS': paths['unknown_fps'],
        'UNKNOWN_CLIPS': paths['unknown_clips'],
        'UNKNOWN_SEP': paths['unknown_faces'],
    }
    
    return constants


# If used directly
if __name__ == "__main__":
    # Configure logging for direct script execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create or load configuration
    config_manager = get_config()
    
    # Print some information
    print("Configuration file:", config_manager.config_path)
    
    for split in ['train', 'test', 'unknown']:
        paths = config_manager.get_paths(split)
        print(f"\n{split.upper()} PATHS:")
        for key, value in paths.items():
            print(f"  {key}: {value}")
    
    # For backwards compatibility
    constants = get_constants()
    print("\nCONSTANTS (for backwards compatibility):")
    for key, value in constants.items():
        print(f"  {key}: {value}")
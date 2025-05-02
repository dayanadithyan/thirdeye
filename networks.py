import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Conv3D, MaxPooling3D, 
    BatchNormalization, Dropout, Reshape, Concatenate, 
    LeakyReLU, Add, GlobalAveragePooling3D, TimeDistributed,
    LayerNormalization, MultiHeadAttention, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# Import our configuration system
from config_manager import ConfigManager, get_config

"""
Neural network architectures for the Thirdeye deepfake detection system.
Implements various 3D CNN architectures and modern variants for video classification.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thirdeye_networks.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Networks")


class NetworkFactory:
    """Factory class for creating neural network models."""
    
    AVAILABLE_ARCHITECTURES = [
        # Legacy architectures
        'providence_v1', 'providence_v2', 'odin_v1', 'odin_v2', 'horus',
        # Modern architectures
        'resnet3d_18', 'resnet3d_34', 'resnet3d_50',
        'slowfast', 'efficient3d', 'vivit',
    ]
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the network factory.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or get_config()
        self.mixed_precision = False
        
        # Check if GPUs are available
        self._check_gpus()
    
    def _check_gpus(self) -> None:
        """Check for available GPUs and configure TensorFlow accordingly."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                logger.info(f"  {gpu.name}")
            
            # Configure memory growth to avoid allocating all GPU memory at once
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Memory growth enabled for all GPUs")
            except RuntimeError as e:
                logger.error(f"Error configuring GPU memory growth: {e}")
            
            # Enable mixed precision for better performance on compatible GPUs
            if len(gpus) > 0:
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    self.mixed_precision = True
                    logger.info("Mixed precision training enabled")
                except Exception as e:
                    logger.error(f"Could not enable mixed precision: {e}")
        else:
            logger.info("No GPUs found, using CPU for training")
    
    def create_model(self, architecture: str, input_shape: Tuple[int, int, int, int], 
                    **kwargs) -> Model:
        """
        Create a model with the specified architecture.
        
        Args:
            architecture: Name of the architecture to create
            input_shape: Shape of the input data (frames, height, width, channels)
            **kwargs: Additional arguments for the model
            
        Returns:
            Keras Model instance
            
        Raises:
            ValueError: If the architecture is not supported
        """
        if architecture not in self.AVAILABLE_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                           f"Available architectures: {self.AVAILABLE_ARCHITECTURES}")
        
        # Legacy architectures
        if architecture in ['providence_v1', 'providence_v2', 'odin_v1', 'odin_v2', 'horus']:
            return self._create_legacy_model(architecture, input_shape, **kwargs)
        
        # Modern architectures
        if architecture.startswith('resnet3d'):
            model_size = int(architecture.split('_')[1])
            return self._create_resnet3d(input_shape, model_size, **kwargs)
        
        if architecture == 'slowfast':
            return self._create_slowfast(input_shape, **kwargs)
        
        if architecture == 'efficient3d':
            return self._create_efficient3d(input_shape, **kwargs)
        
        if architecture == 'vivit':
            return self._create_vivit(input_shape, **kwargs)
    
    def _create_legacy_model(self, architecture: str, input_shape: Tuple[int, int, int, int], 
                           **kwargs) -> Model:
        """
        Create one of the legacy models from the original Thirdeye system.
        
        Args:
            architecture: Name of the legacy architecture
            input_shape: Shape of the input data
            **kwargs: Additional arguments
            
        Returns:
            Keras Model instance
        """
        # Get parameters for the specified architecture
        params = self.config.get_network_params(architecture)
        
        # Extract parameters or use defaults from kwargs
        filters1 = kwargs.get('filters1', params.get('filters1', 8))
        filters2 = kwargs.get('filters2', params.get('filters2', 16))
        conv2 = kwargs.get('conv2', params.get('conv2', True))
        conv4 = kwargs.get('conv4', params.get('conv4', True))
        nodes_1 = kwargs.get('nodes_1', params.get('nodes_1', 32))
        nodes_2 = kwargs.get('nodes_2', params.get('nodes_2', 16))
        leaky = kwargs.get('leaky', params.get('leaky', False))
        
        # Define the input layer
        input_layer = Input(shape=input_shape)
        
        # First convolutional block
        x = Conv3D(filters=filters1, kernel_size=(3, 3, 3), 
                  activation='relu', padding='same')(input_layer)
        
        # Second convolutional layer if needed
        if conv2:
            if input_shape[0] > 5:  # If we have enough frames
                x = Conv3D(filters=filters2, kernel_size=(3, 3, 3), 
                          activation='relu', padding='same')(x)
            else:
                x = Conv3D(filters=filters2, kernel_size=(1, 3, 3), 
                          activation='relu', padding='same')(x)
        
        # First pooling layer
        if input_shape[0] > 5:
            x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        else:
            x = MaxPooling3D(pool_size=(1, 2, 2))(x)
        
        # Third convolutional layer
        if input_shape[0] > 8:
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), 
                      activation='relu', padding='same')(x)
        else:
            x = Conv3D(filters=32, kernel_size=(1, 3, 3), 
                      activation='relu', padding='same')(x)
        
        # Fourth convolutional layer if needed
        if conv4:
            if input_shape[0] > 11:
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), 
                          activation='relu', padding='same')(x)
            else:
                x = Conv3D(filters=64, kernel_size=(1, 3, 3), 
                          activation='relu', padding='same')(x)
        
        # Second pooling layer
        if input_shape[0] > 14:
            x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        else:
            x = MaxPooling3D(pool_size=(1, 2, 2))(x)
        
        # Normalization and flattening
        x = BatchNormalization()(x)
        x = Flatten()(x)
        
        # First dense layer
        x = Dense(units=nodes_1, activation='relu')(x)
        x = Dropout(0.4)(x)
        
        # Second dense layer or LeakyReLU
        if leaky:
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = Dense(units=nodes_2, activation='relu')(x)
        
        # Final dropout and output layer
        x = Dropout(0.4)(x)
        output_layer = Dense(2, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(
            loss=CategoricalCrossentropy(),
            optimizer=Adadelta(learning_rate=0.1),
            metrics=['accuracy']
        )
        
        return model
    
    def _create_resnet3d(self, input_shape: Tuple[int, int, int, int], 
                        model_size: int = 50, **kwargs) -> Model:
        """
        Create a 3D ResNet model.
        
        Args:
            input_shape: Shape of the input data
            model_size: Size of the ResNet model (18, 34, 50, etc.)
            **kwargs: Additional arguments
            
        Returns:
            Keras Model instance
        """
        # Define the ResNet block configurations
        if model_size == 18:
            block_config = [2, 2, 2, 2]
            filters = [64, 128, 256, 512]
        elif model_size == 34:
            block_config = [3, 4, 6, 3]
            filters = [64, 128, 256, 512]
        elif model_size == 50:
            block_config = [3, 4, 6, 3]
            filters = [64, 128, 256, 512]
            # Bottleneck architecture
            bottleneck = True
        else:
            raise ValueError(f"Unsupported ResNet size: {model_size}")
        
        bottleneck = kwargs.get('bottleneck', model_size >= 50)
        
        # Define the input layer
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv3D(filters=64, kernel_size=(3, 7, 7), strides=(1, 2, 2), 
                  padding='same', use_bias=False, 
                  kernel_regularizer=l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Max pooling after initial convolution
        x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
        
        # ResNet blocks
        for i, blocks in enumerate(block_config):
            for j in range(blocks):
                stride = (1, 2, 2) if i > 0 and j == 0 else (1, 1, 1)
                
                if bottleneck:
                    # Bottleneck block for ResNet50+
                    x = self._bottleneck_block(x, filters[i], stride)
                else:
                    # Basic block for ResNet18/34
                    x = self._basic_block(x, filters[i], stride)
        
        # Global pooling
        x = GlobalAveragePooling3D()(x)
        
        # Dense layers
        x = Dense(units=512, activation='relu', 
                 kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)
        outputs = Dense(units=2, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use a lower learning rate for deeper models
        lr = 1e-4 if model_size >= 50 else 1e-3
        
        # Compile model
        model.compile(
            loss=CategoricalCrossentropy(),
            optimizer=Adam(learning_rate=lr),
            metrics=['accuracy']
        )
        
        return model
    
    def _basic_block(self, x, filters: int, strides: Tuple[int, int, int]) -> tf.Tensor:
        """
        Create a basic ResNet block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            strides: Stride dimensions
            
        Returns:
            Output tensor
        """
        identity = x
        
        # First convolution
        x = Conv3D(filters=filters, kernel_size=3, strides=strides, 
                  padding='same', use_bias=False, 
                  kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second convolution
        x = Conv3D(filters=filters, kernel_size=3, strides=(1, 1, 1), 
                  padding='same', use_bias=False, 
                  kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        
        # Skip connection
        if strides != (1, 1, 1) or identity.shape[-1] != filters:
            identity = Conv3D(filters=filters, kernel_size=1, strides=strides, 
                            padding='same', use_bias=False, 
                            kernel_regularizer=l2(1e-4))(identity)
            identity = BatchNormalization()(identity)
        
        # Add skip connection
        x = Add()([x, identity])
        x = Activation('relu')(x)
        
        return x
    
    def _bottleneck_block(self, x, filters: int, strides: Tuple[int, int, int]) -> tf.Tensor:
        """
        Create a bottleneck ResNet block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            strides: Stride dimensions
            
        Returns:
            Output tensor
        """
        identity = x
        
        # 1x1 convolution to reduce dimensions
        x = Conv3D(filters=filters, kernel_size=1, strides=(1, 1, 1), 
                  padding='same', use_bias=False, 
                  kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 3x3 convolution
        x = Conv3D(filters=filters, kernel_size=3, strides=strides, 
                  padding='same', use_bias=False, 
                  kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 1x1 convolution to expand dimensions
        x = Conv3D(filters=filters * 4, kernel_size=1, strides=(1, 1, 1), 
                  padding='same', use_bias=False, 
                  kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        
        # Skip connection
        if strides != (1, 1, 1) or identity.shape[-1] != filters * 4:
            identity = Conv3D(filters=filters * 4, kernel_size=1, strides=strides, 
                            padding='same', use_bias=False, 
                            kernel_regularizer=l2(1e-4))(identity)
            identity = BatchNormalization()(identity)
        
        # Add skip connection
        x = Add()([x, identity])
        x = Activation('relu')(x)
        
        return x
    
    def _create_slowfast(self, input_shape: Tuple[int, int, int, int], **kwargs) -> Model:
        """
        Create a SlowFast network for video classification.
        Based on "SlowFast Networks for Video Recognition" by Feichtenhofer et al.
        
        Args:
            input_shape: Shape of the input data
            **kwargs: Additional arguments
            
        Returns:
            Keras Model instance
        """
        alpha = kwargs.get('alpha', 8)  # Sampling rate ratio between slow and fast paths
        beta = kwargs.get('beta', 1/8)  # Channel ratio between slow and fast paths
        
        frames, height, width, channels = input_shape
        
        # Ensure we have enough frames for both pathways
        if frames < alpha:
            raise ValueError(f"Input must have at least {alpha} frames for SlowFast network")
        
        # Create the slow pathway (lower frame rate, more channels)
        slow_frames = frames // alpha
        slow_input_shape = (slow_frames, height, width, channels)
        slow_input = Input(slow_input_shape)
        
        # Create the fast pathway (higher frame rate, fewer channels)
        fast_input_shape = (frames, height, width, channels)
        fast_input = Input(fast_input_shape)
        
        # Build the slow pathway
        slow = Conv3D(filters=64, kernel_size=(1, 7, 7), strides=(1, 2, 2), 
                     padding='same', use_bias=False)(slow_input)
        slow = BatchNormalization()(slow)
        slow = Activation('relu')(slow)
        slow = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(slow)
        
        # Build the fast pathway with fewer channels
        fast_filters = int(64 * beta)
        fast = Conv3D(filters=fast_filters, kernel_size=(5, 7, 7), strides=(1, 2, 2), 
                     padding='same', use_bias=False)(fast_input)
        fast = BatchNormalization()(fast)
        fast = Activation('relu')(fast)
        fast = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(fast)
        
        # ResNet blocks for the slow pathway
        slow_filters = [64, 128, 256, 512]
        fast_filters = [int(f * beta) for f in slow_filters]
        
        # Four ResNet stages
        for i, filters in enumerate(slow_filters):
            # Slow pathway
            for j in range(2):  # 2 blocks per stage
                stride = (1, 2, 2) if i > 0 and j == 0 else (1, 1, 1)
                slow = self._basic_block(slow, filters, stride)
            
            # Fast pathway
            for j in range(2):  # 2 blocks per stage
                stride = (1, 2, 2) if i > 0 and j == 0 else (1, 1, 1)
                fast = self._basic_block(fast, fast_filters[i], stride)
        
        # Global pooling
        slow = GlobalAveragePooling3D()(slow)
        fast = GlobalAveragePooling3D()(fast)
        
        # Concatenate pathways
        x = Concatenate()([slow, fast])
        
        # Dense layers
        x = Dense(units=512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)
        outputs = Dense(units=2, activation='softmax')(x)
        
        # Create model with two inputs
        model = Model(inputs=[slow_input, fast_input], outputs=outputs)
        
        # Compile model
        model.compile(
            loss=CategoricalCrossentropy(),
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )
        
        return model
    
    def _create_efficient3d(self, input_shape: Tuple[int, int, int, int], **kwargs) -> Model:
        """
        Create a 3D EfficientNet model.
        Based on the 2D EfficientNet architecture but extended to 3D.
        
        Args:
            input_shape: Shape of the input data
            **kwargs: Additional arguments
            
        Returns:
            Keras Model instance
        """
        width_coefficient = kwargs.get('width_coefficient', 1.0)
        depth_coefficient = kwargs.get('depth_coefficient', 1.0)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        
        # Define the MBConv block configurations (stage, repeats, filters)
        block_configs = [
            # stage, repeats, filters
            (1, 1, 16),
            (2, 2, 24),
            (3, 2, 40),
            (4, 3, 80),
            (5, 3, 112),
            (6, 4, 192),
            (7, 1, 320),
        ]
        
        # Scale the depth (repeats) based on the depth_coefficient
        def round_repeats(repeats):
            return int(tf.math.ceil(depth_coefficient * repeats))
        
        # Scale the filters based on the width_coefficient
        def round_filters(filters):
            return int(tf.math.ceil(width_coefficient * filters))
        
        # Define the input layer
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv3D(filters=round_filters(32), kernel_size=(3, 3, 3), 
                  strides=(1, 2, 2), padding='same', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        
        # Build MBConv blocks
        for stage, repeats, filters in block_configs:
            repeats = round_repeats(repeats)
            filters = round_filters(filters)
            
            # First block might use different strides
            strides = (1, 2, 2) if stage > 1 else (1, 1, 1)
            x = self._mb_conv_block(x, filters, strides, expansion_factor=6)
            
            # Additional blocks with stride 1
            for _ in range(repeats - 1):
                x = self._mb_conv_block(x, filters, (1, 1, 1), expansion_factor=6)
        
        # Final convolution
        x = Conv3D(filters=round_filters(1280), kernel_size=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        
        # Global pooling
        x = GlobalAveragePooling3D()(x)
        
        # Dropout
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(units=2, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            loss=CategoricalCrossentropy(),
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )
        
        return model
    
    def _mb_conv_block(self, x, filters: int, strides: Tuple[int, int, int], 
                      expansion_factor: int = 6) -> tf.Tensor:
        """
        Create a MBConv (Mobile Inverted Bottleneck) block for EfficientNet.
        
        Args:
            x: Input tensor
            filters: Number of output filters
            strides: Stride dimensions
            expansion_factor: Expansion factor for the block
            
        Returns:
            Output tensor
        """
        input_filters = x.shape[-1]
        expanded_filters = int(input_filters * expansion_factor)
        
        # Skip connection
        skip_connection = (strides == (1, 1, 1) and input_filters == filters)
        
        # Expansion phase
        if expansion_factor > 1:
            expand = Conv3D(filters=expanded_filters, kernel_size=1, 
                           padding='same', use_bias=False)(x)
            expand = BatchNormalization()(expand)
            expand = Activation('swish')(expand)
        else:
            expand = x
        
        # Depthwise convolution
        depthwise = Conv3D(filters=expanded_filters, kernel_size=3, 
                          strides=strides, padding='same', 
                          use_bias=False, groups=expanded_filters)(expand)
        depthwise = BatchNormalization()(depthwise)
        depthwise = Activation('swish')(depthwise)
        
        # Squeeze and excitation
        squeeze = GlobalAveragePooling3D()(depthwise)
        squeeze = Dense(units=int(input_filters / 4), activation='swish')(squeeze)
        squeeze = Dense(units=expanded_filters, activation='sigmoid')(squeeze)
        
        # Reshape for broadcasting
        squeeze = Reshape((1, 1, 1, expanded_filters))(squeeze)
        scale = depthwise * squeeze
        
        # Projection phase
        project = Conv3D(filters=filters, kernel_size=1, padding='same', use_bias=False)(scale)
        project = BatchNormalization()(project)
        
        # Skip connection
        if skip_connection:
            return Add()([x, project])
        
        return project
    
    def _create_vivit(self, input_shape: Tuple[int, int, int, int], **kwargs) -> Model:
        """
        Create a Video Vision Transformer (ViViT) model.
        Based on "ViViT: A Video Vision Transformer" by Arnab et al.
        
        Args:
            input_shape: Shape of the input data
            **kwargs: Additional arguments
            
        Returns:
            Keras Model instance
        """
        # Configuration
        num_heads = kwargs.get('num_heads', 8)
        num_layers = kwargs.get('num_layers', 12)
        embed_dim = kwargs.get('embed_dim', 768)
        mlp_dim = kwargs.get('mlp_dim', 3072)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        patch_size = kwargs.get('patch_size', (2, 16, 16))
        
        frames, height, width, channels = input_shape
        
        # Calculate the number of patches
        num_patches_temporal = frames // patch_size[0]
        num_patches_height = height // patch_size[1]
        num_patches_width = width // patch_size[2]
        num_patches = num_patches_temporal * num_patches_height * num_patches_width
        
        # Define the input layer
        inputs = Input(shape=input_shape)
        
        # Create patches
        patches = self._extract_patches_3d(inputs, patch_size)
        
        # Patch embedding
        x = Conv3D(filters=embed_dim, kernel_size=1, strides=1)(patches)
        x = Reshape((num_patches, embed_dim))(x)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
        x = x + pos_embedding
        
        # Transformer encoder
        for _ in range(num_layers):
            # Layer normalization 1
            norm1 = LayerNormalization(epsilon=1e-6)(x)
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
            )(norm1, norm1)
            
            # Skip connection 1
            x = Add()([attention_output, x])
            
            # Layer normalization 2
            norm2 = LayerNormalization(epsilon=1e-6)(x)
            
            # MLP
            mlp = Dense(mlp_dim, activation='gelu')(norm2)
            mlp = Dropout(dropout_rate)(mlp)
            mlp = Dense(embed_dim)(mlp)
            mlp = Dropout(dropout_rate)(mlp)
            
            # Skip connection 2
            x = Add()([mlp, x])
        
        # Layer normalization
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Classification head
        x = Dense(units=512, activation='tanh')(x)
        outputs = Dense(units=2, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            loss=CategoricalCrossentropy(),
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )
        
        return model
    
    def _extract_patches_3d(self, x, patch_size: Tuple[int, int, int]) -> tf.Tensor:
        """
        Extract 3D patches from the input tensor.
        
        Args:
            x: Input tensor
            patch_size: Size of each patch (temporal, height, width)
            
        Returns:
            Tensor with extracted patches
        """
        # Get input dimensions
        frames, height, width, channels = x.shape[1:]
        
        # Calculate number of patches
        num_patches_temporal = frames // patch_size[0]
        num_patches_height = height // patch_size[1]
        num_patches_width = width // patch_size[2]
        
        # Reshape input to extract patches
        x = tf.reshape(
            x, 
            (-1, 
             num_patches_temporal, patch_size[0], 
             num_patches_height, patch_size[1], 
             num_patches_width, patch_size[2], 
             channels)
        )
        
        # Transpose and reshape to get patches
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
        x = tf.reshape(
            x, 
            (-1, 
             num_patches_temporal * num_patches_height * num_patches_width, 
             patch_size[0] * patch_size[1] * patch_size[2] * channels)
        )
        
        return x


class DeepfakeDetectionNetwork:
    """Main class for managing deepfake detection networks."""
    
    def __init__(self, config_path: str = 'config.yaml',
                architecture: str = 'odin_v1',
                model_dir: str = None):
        """
        Initialize the deepfake detection network.
        
        Args:
            config_path: Path to the configuration file
            architecture: Name of the architecture to use
            model_dir: Directory to save models
        """
        self.config_manager = get_config(config_path)
        self.factory = NetworkFactory(self.config_manager)
        
        self.architecture = architecture
        self.model_dir = model_dir or self.config_manager.config['data_paths']['models_dir']
        self.model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def build_model(self, input_shape: Tuple[int, int, int, int], **kwargs) -> Model:
        """
        Build a model with the specified architecture.
        
        Args:
            input_shape: Shape of the input data
            **kwargs: Additional arguments for the model
            
        Returns:
            Keras Model instance
        """
        self.model = self.factory.create_model(self.architecture, input_shape, **kwargs)
        return self.model
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
             x_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None,
             epochs: int = None, batch_size: int = None,
             callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            callbacks: Additional callbacks for training
            
        Returns:
            Training history
            
        Raises:
            ValueError: If the model is not built
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Get parameters from config
        model_params = self.config_manager.get_model_params()
        network_params = self.config_manager.get_network_params(self.architecture)
        
        epochs = epochs or network_params.get('epochs', model_params.get('epochs', 10))
        batch_size = batch_size or model_params.get('batch_size', 32)
        
        # Prepare callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Train the model
        if x_val is not None and y_val is not None:
            history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Use validation split if validation data is not provided
            validation_split = model_params.get('validation_split', 0.2)
            history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
        
        # Save the history and model
        self.save_history(history)
        self.save_model()
        
        return history
    
    def _get_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Get default callbacks for training.
        
        Returns:
            List of callbacks
        """
        model_path = os.path.join(self.model_dir, f"{self.architecture}_best.h5")
        log_path = os.path.join(self.model_dir, f"{self.architecture}_logs")
        csv_path = os.path.join(self.model_dir, f"{self.architecture}_training.csv")
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(log_dir=log_path),
            # CSV logging
            CSVLogger(csv_path)
        ]
        
        return callbacks
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            x_test: Test data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
            
        Raises:
            ValueError: If the model is not built
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Evaluate the model
        results = self.model.evaluate(x_test, y_test, verbose=1)
        
        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            x: Input data
            
        Returns:
            Predictions
            
        Raises:
            ValueError: If the model is not built
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.predict(x)
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the model.
        
        Args:
            filepath: Path to save the model (optional)
            
        Raises:
            ValueError: If the model is not built
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{self.architecture}.h5")
        
        # Save the model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        json_filepath = os.path.join(self.model_dir, f"{self.architecture}_architecture.json")
        with open(json_filepath, 'w') as f:
            f.write(model_json)
        logger.info(f"Model architecture saved to {json_filepath}")
    
    def load_model(self, filepath: Optional[str] = None) -> None:
        """
        Load the model.
        
        Args:
            filepath: Path to the saved model (optional)
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{self.architecture}.h5")
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file {filepath} not found")
            return
        
        # Load the model
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def save_history(self, history: tf.keras.callbacks.History) -> None:
        """
        Save training history.
        
        Args:
            history: Training history
        """
        history_dict = {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy']
        }
        
        if 'val_loss' in history.history:
            history_dict['val_loss'] = history.history['val_loss']
            history_dict['val_accuracy'] = history.history['val_accuracy']
        
        # Save as numpy file
        history_path = os.path.join(self.model_dir, f"{self.architecture}_history.npz")
        np.savez(history_path, **history_dict)
        logger.info(f"Training history saved to {history_path}")
    
    def load_history(self) -> Dict[str, np.ndarray]:
        """
        Load training history.
        
        Returns:
            Dictionary of training history
        """
        history_path = os.path.join(self.model_dir, f"{self.architecture}_history.npz")
        
        if not os.path.exists(history_path):
            logger.warning(f"History file {history_path} not found")
            return {}
        
        # Load history
        history_data = np.load(history_path)
        history_dict = {key: history_data[key] for key in history_data.files}
        
        return history_dict
    
    def plot_training_history(self, save_fig: bool = True) -> None:
        """
        Plot the training history.
        
        Args:
            save_fig: Whether to save the figure
        """
        history = self.load_history()
        
        if not history:
            logger.warning("No history data found")
            return
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training')
        if 'val_accuracy' in history:
            ax1.plot(history['val_accuracy'], label='Validation')
        ax1.set_title(f'{self.architecture} Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['loss'], label='Training')
        if 'val_loss' in history:
            ax2.plot(history['val_loss'], label='Validation')
        ax2.set_title(f'{self.architecture} Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_fig:
            figures_dir = self.config_manager.config['data_paths']['figures_dir']
            os.makedirs(figures_dir, exist_ok=True)
            fig_path = os.path.join(figures_dir, f"{self.architecture}_history.png")
            plt.savefig(fig_path)
            logger.info(f"Training history plot saved to {fig_path}")
        
        plt.show()
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model summary.
        
        Returns:
            Model summary string
            
        Raises:
            ValueError: If the model is not built
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Capture the model summary in a string
        string_io = tf.io.StringIO()
        self.model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        summary_string = string_io.getvalue()
        string_io.close()
        
        return summary_string


# For backwards compatibility
class Network:
    """Legacy Network class for backwards compatibility."""
    
    def __init__(self, summary: bool = False, name: str = ''):
        """
        Initialize the legacy network class.
        
        Args:
            summary: Whether to print model summary
            name: Name of the model to load
        """
        self.summary = summary
        self.model = None
        self.network = DeepfakeDetectionNetwork(architecture=name if name else 'odin_v1')
        
        if name:
            self.load_network(name)
    
    def load_network(self, name: str, xtrain=[], ytrain=[], 
                   xtest=[], ytest=[], train: bool = False) -> None:
        """
        Load a network model.
        
        Args:
            name: Name of the model
            xtrain: Training data
            ytrain: Training labels
            xtest: Test data
            ytest: Test labels
            train: Whether to train the model
        """
        self.network.architecture = name
        
        if train and len(xtrain) > 0 and len(ytrain) > 0:
            # Build and train the model
            input_shape = xtrain[0].shape
            self.network.build_model(input_shape)
            
            if self.summary:
                print(self.network.get_model_summary())
            
            if len(xtest) > 0 and len(ytest) > 0:
                self.network.train(xtrain, ytrain, xtest, ytest)
            else:
                self.network.train(xtrain, ytrain)
        else:
            # Try to load the model
            self.network.load_model()
            
            if self.network.model is None:
                logger.warning(f"No saved model found for {name}")
            elif self.summary:
                print(self.network.get_model_summary())
                print(f"{name.capitalize()} is ready.")
        
        self.model = self.network.model
    
    def set_model(self, name: str, xtrain=[], ytrain=[], train: bool = False) -> None:
        """
        Set the current model.
        
        Args:
            name: Name of the model
            xtrain: Training data
            ytrain: Training labels
            train: Whether to train the model
        """
        self.load_network(name, xtrain, ytrain, train=train)
    
    def get_model(self) -> tf.keras.Model:
        """
        Get the current model.
        
        Returns:
            Keras Model instance
        """
        return self.model


# If used directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Thirdeye networks module')
    parser.add_argument('--architecture', type=str, default='odin_v1',
                       choices=NetworkFactory.AVAILABLE_ARCHITECTURES,
                       help='Network architecture to use')
    parser.add_argument('--summary', action='store_true',
                       help='Print model summary')
    
    args = parser.parse_args()
    
    # Create sample input shape
    sample_input_shape = (20, 100, 100, 3)  # (frames, height, width, channels)
    
    # Create network
    network = DeepfakeDetectionNetwork(architecture=args.architecture)
    model = network.build_model(sample_input_shape)
    
    if args.summary:
        print(network.get_model_summary())
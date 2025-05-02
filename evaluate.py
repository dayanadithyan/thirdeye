import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from itertools import cycle
import datetime
import json
from tqdm import tqdm

# Import our configuration and utilities
from config_manager import ConfigManager, get_config
from utilities import FileSystemUtilities

"""
Evaluation module for Thirdeye deepfake detection system.
Provides comprehensive evaluation metrics and visualizations for model performance.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thirdeye_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evaluation")


class EvaluationMetrics:
    """Class for calculating and storing evaluation metrics."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_score: Optional[np.ndarray] = None,
                class_names: Optional[List[str]] = None):
        """
        Initialize the evaluation metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted labels (one-hot encoded)
            y_score: Prediction probabilities (optional)
            class_names: Names of the classes (optional)
        """
        # Store inputs
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score if y_score is not None else y_pred
        
        # Default class names
        self.class_names = class_names or ['Real', 'Deepfake']
        self.n_classes = len(self.class_names)
        
        # Convert one-hot to class indices if necessary
        self.y_true_indices = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        self.y_pred_indices = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        
        # Calculate metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate all evaluation metrics."""
        # Basic metrics
        self.accuracy = accuracy_score(self.y_true_indices, self.y_pred_indices)
        
        # Confusion matrix
        self.confusion_matrix = confusion_matrix(
            self.y_true_indices, self.y_pred_indices, labels=range(self.n_classes)
        )
        
        # Per-class metrics
        self.precision = precision_score(
            self.y_true_indices, self.y_pred_indices, 
            average=None, labels=range(self.n_classes)
        )
        self.recall = recall_score(
            self.y_true_indices, self.y_pred_indices, 
            average=None, labels=range(self.n_classes)
        )
        self.f1 = f1_score(
            self.y_true_indices, self.y_pred_indices, 
            average=None, labels=range(self.n_classes)
        )
        
        # Macro-averaged metrics
        self.macro_precision = precision_score(
            self.y_true_indices, self.y_pred_indices, average='macro'
        )
        self.macro_recall = recall_score(
            self.y_true_indices, self.y_pred_indices, average='macro'
        )
        self.macro_f1 = f1_score(
            self.y_true_indices, self.y_pred_indices, average='macro'
        )
        
        # Calculate ROC and AUC if we have probability scores
        self.fpr = {}
        self.tpr = {}
        self.roc_auc = {}
        
        for i in range(self.n_classes):
            # Get true labels and scores for this class
            if self.y_true.ndim > 1:
                true_class = self.y_true[:, i]
            else:
                true_class = (self.y_true_indices == i).astype(int)
            
            if self.y_score.ndim > 1:
                score_class = self.y_score[:, i]
            else:
                score_class = (self.y_score == i).astype(float)
            
            # Calculate ROC curve and AUC
            self.fpr[i], self.tpr[i], _ = roc_curve(true_class, score_class)
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
        
        # Calculate macro-average ROC curve
        # First, combine all FPR points
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))
        
        # Then interpolate TPR at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])
        
        # Average and compute AUC
        mean_tpr /= self.n_classes
        self.fpr['macro'] = all_fpr
        self.tpr['macro'] = mean_tpr
        self.roc_auc['macro'] = auc(self.fpr['macro'], self.tpr['macro'])
        
        # Calculate precision-recall curve
        self.precision_curve = {}
        self.recall_curve = {}
        self.average_precision = {}
        
        for i in range(self.n_classes):
            # Get true labels and scores for this class
            if self.y_true.ndim > 1:
                true_class = self.y_true[:, i]
            else:
                true_class = (self.y_true_indices == i).astype(int)
            
            if self.y_score.ndim > 1:
                score_class = self.y_score[:, i]
            else:
                score_class = (self.y_score == i).astype(float)
            
            # Calculate precision-recall curve
            self.precision_curve[i], self.recall_curve[i], _ = precision_recall_curve(
                true_class, score_class
            )
            self.average_precision[i] = average_precision_score(true_class, score_class)
        
        # Calculate macro-average precision-recall curve
        self.average_precision['macro'] = sum(self.average_precision.values()) / self.n_classes
    
    def get_classification_report(self) -> str:
        """
        Get a text classification report.
        
        Returns:
            Classification report as string
        """
        return classification_report(
            self.y_true_indices, self.y_pred_indices, 
            target_names=self.class_names, digits=4
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        summary = {
            'accuracy': float(self.accuracy),
            'macro_precision': float(self.macro_precision),
            'macro_recall': float(self.macro_recall),
            'macro_f1': float(self.macro_f1),
            'macro_auc': float(self.roc_auc['macro']),
            'per_class': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            summary['per_class'][class_name] = {
                'precision': float(self.precision[i]),
                'recall': float(self.recall[i]),
                'f1': float(self.f1[i]),
                'auc': float(self.roc_auc[i]),
                'average_precision': float(self.average_precision[i])
            }
        
        return summary


class Evaluator:
    """Class for evaluating deepfake detection models."""
    
    def __init__(self, model=None, config_manager: Optional[ConfigManager] = None,
                show_plots: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            model: TensorFlow model to evaluate (optional)
            config_manager: Configuration manager instance
            show_plots: Whether to show plots or just save them
        """
        self.model = model
        self.config = config_manager or get_config()
        self.show_plots = show_plots
        
        # Create output directories
        self.figures_dir = self.config.config['data_paths']['figures_dir']
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set default plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Initialize TensorBoard
        self.tensorboard_dir = os.path.join(
            self.figures_dir, 'tensorboard', 
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.tensorboard_callback = TensorBoard(
            log_dir=self.tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        )
    
    def plot_training_history(self, history: Union[Dict, tf.keras.callbacks.History], 
                            model_name: str = '') -> plt.Figure:
        """
        Plot training history metrics.
        
        Args:
            history: Training history (dictionary or History object)
            model_name: Name of the model
            
        Returns:
            Matplotlib figure
        """
        # Convert History object to dictionary if needed
        if isinstance(history, tf.keras.callbacks.History):
            history_dict = history.history
        else:
            history_dict = history
        
        # Extract metrics
        metrics = []
        val_metrics = []
        
        for key in history_dict.keys():
            if not key.startswith('val_'):
                metrics.append(key)
                val_key = f'val_{key}'
                if val_key in history_dict:
                    val_metrics.append(val_key)
        
        # Calculate number of subplots
        n_metrics = len(metrics)
        n_cols = min(n_metrics, 3)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        
        # Handle case with single subplot
        if n_metrics == 1:
            axes = [axes]
        # Flatten axes array for multiple subplots
        elif n_metrics > 1:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training metric
            ax.plot(history_dict[metric], label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                ax.plot(history_dict[val_metric], label=f'Validation {metric}')
            
            # Set labels and title
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{model_name} {metric.capitalize()}')
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.figures_dir, f'{model_name}_history.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {fig_path}")
        
        # Show the figure if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = '', 
                            class_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            class_names: Names of the classes
            
        Returns:
            Matplotlib figure
        """
        # Default class names
        if class_names is None:
            class_names = ['Real', 'Deepfake']
        
        # Convert one-hot to class indices if necessary
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_title(f'{model_name} Confusion Matrix (Counts)')
        
        # Plot percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_title(f'{model_name} Confusion Matrix (Percentages)')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.figures_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {fig_path}")
        
        # Show the figure if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                      model_name: str = '', 
                      class_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot ROC curves.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_score: Prediction probabilities
            model_name: Name of the model
            class_names: Names of the classes
            
        Returns:
            Matplotlib figure
        """
        # Default class names
        if class_names is None:
            class_names = ['Real', 'Deepfake']
        
        # Create metrics object
        metrics = EvaluationMetrics(y_true, y_score, y_score, class_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
        
        for i, color, class_name in zip(range(len(class_names)), colors, class_names):
            plt.plot(
                metrics.fpr[i], metrics.tpr[i], color=color, lw=2,
                label=f'ROC curve of {class_name} (AUC = {metrics.roc_auc[i]:.3f})'
            )
        
        # Plot macro-average ROC curve
        plt.plot(
            metrics.fpr['macro'], metrics.tpr['macro'], color='purple', lw=2, linestyle=':',
            label=f'Macro-average ROC curve (AUC = {metrics.roc_auc["macro"]:.3f})'
        )
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set labels and limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')
        
        # Save the figure
        fig_path = os.path.join(self.figures_dir, f'{model_name}_roc_curve.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve plot saved to {fig_path}")
        
        # Show the figure if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                                  model_name: str = '', 
                                  class_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot precision-recall curves.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_score: Prediction probabilities
            model_name: Name of the model
            class_names: Names of the classes
            
        Returns:
            Matplotlib figure
        """
        # Default class names
        if class_names is None:
            class_names = ['Real', 'Deepfake']
        
        # Create metrics object
        metrics = EvaluationMetrics(y_true, y_score, y_score, class_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot precision-recall curve for each class
        colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
        
        for i, color, class_name in zip(range(len(class_names)), colors, class_names):
            plt.plot(
                metrics.recall_curve[i], metrics.precision_curve[i], color=color, lw=2,
                label=f'Precision-Recall curve of {class_name} (AP = {metrics.average_precision[i]:.3f})'
            )
        
        # Set labels and limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        # Save the figure
        fig_path = os.path.join(self.figures_dir, f'{model_name}_precision_recall_curve.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-recall curve plot saved to {fig_path}")
        
        # Show the figure if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def evaluate_model(self, x_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = '', batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            x_test: Test data
            y_test: Test labels
            model_name: Name of the model
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary with evaluation metrics
            
        Raises:
            ValueError: If no model is available
        """
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        logger.info(f"Evaluating model {model_name} on {len(x_test)} samples")
        
        # Start timer
        start_time = datetime.datetime.now()
        
        # Get predictions
        y_score = self.model.predict(x_test, batch_size=batch_size)
        y_pred = np.argmax(y_score, axis=1)
        
        # Convert y_test to indices if one-hot encoded
        if y_test.ndim > 1:
            y_test_indices = np.argmax(y_test, axis=1)
        else:
            y_test_indices = y_test
        
        # Calculate evaluation time
        evaluation_time = (datetime.datetime.now() - start_time).total_seconds()
        predictions_per_second = len(x_test) / evaluation_time
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s "
                   f"({predictions_per_second:.2f} predictions/s)")
        
        # Create metrics object
        metrics = EvaluationMetrics(y_test, y_score, y_score)
        
        # Get summary
        summary = metrics.get_summary()
        summary['evaluation_time'] = evaluation_time
        summary['predictions_per_second'] = predictions_per_second
        
        # Print accuracy
        logger.info(f"Accuracy: {summary['accuracy'] * 100:.2f}%")
        
        # Print classification report
        print(metrics.get_classification_report())
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test_indices, y_pred, model_name)
        
        # Plot ROC curve
        self.plot_roc_curve(y_test, y_score, model_name)
        
        # Plot precision-recall curve
        self.plot_precision_recall_curve(y_test, y_score, model_name)
        
        # Save summary to JSON
        summary_path = os.path.join(self.figures_dir, f'{model_name}_evaluation.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Evaluation summary saved to {summary_path}")
        
        return summary
    
    def analyze_failures(self, x_test: np.ndarray, y_test: np.ndarray,
                        model_name: str = '', 
                        max_examples: int = 10,
                        batch_size: int = 32) -> plt.Figure:
        """
        Analyze and visualize model failures.
        
        Args:
            x_test: Test data
            y_test: Test labels
            model_name: Name of the model
            max_examples: Maximum number of examples to visualize
            batch_size: Batch size for prediction
            
        Returns:
            Matplotlib figure
            
        Raises:
            ValueError: If no model is available
        """
        if self.model is None:
            raise ValueError("No model available for analysis")
        
        # Get predictions
        y_score = self.model.predict(x_test, batch_size=batch_size)
        y_pred = np.argmax(y_score, axis=1)
        
        # Convert y_test to indices if one-hot encoded
        if y_test.ndim > 1:
            y_test_indices = np.argmax(y_test, axis=1)
        else:
            y_test_indices = y_test
        
        # Find misclassified examples
        misclassified = np.where(y_test_indices != y_pred)[0]
        
        if len(misclassified) == 0:
            logger.info("No misclassified examples found")
            return None
        
        # Select examples to visualize
        n_examples = min(len(misclassified), max_examples)
        examples_idx = np.random.choice(misclassified, n_examples, replace=False)
        
        # Extract examples
        examples = x_test[examples_idx]
        true_labels = y_test_indices[examples_idx]
        pred_labels = y_pred[examples_idx]
        confidences = np.max(y_score[examples_idx], axis=1)
        
        # Create figure
        fig, axes = plt.subplots(n_examples, 1, figsize=(10, 4 * n_examples))
        
        # Handle case with single example
        if n_examples == 1:
            axes = [axes]
        
        # Plot each example
        for i in range(n_examples):
            example = examples[i]
            
            # If the example is a video (3D), select middle frame
            if example.ndim == 4:  # (frames, height, width, channels)
                middle_frame = example[example.shape[0] // 2]
            else:
                middle_frame = example
            
            # Convert to uint8 if necessary
            if middle_frame.dtype != np.uint8:
                if np.max(middle_frame) <= 1.0:
                    middle_frame = (middle_frame * 255).astype(np.uint8)
                else:
                    middle_frame = middle_frame.astype(np.uint8)
            
            # Plot frame
            axes[i].imshow(middle_frame)
            
            # Set title
            true_class = 'Real' if true_labels[i] == 0 else 'Deepfake'
            pred_class = 'Real' if pred_labels[i] == 0 else 'Deepfake'
            title = f"True: {true_class}, Predicted: {pred_class}, Confidence: {confidences[i]:.2f}"
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.figures_dir, f'{model_name}_failure_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Failure analysis saved to {fig_path}")
        
        # Show the figure if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def cross_validate(self, x: np.ndarray, y: np.ndarray, model_builder: Callable,
                      n_splits: int = 5, model_name: str = '',
                      batch_size: int = 32, epochs: int = 10) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            x: Data
            y: Labels
            model_builder: Function that returns a new model
            n_splits: Number of cross-validation splits
            model_name: Name of the model
            batch_size: Batch size for training
            epochs: Number of epochs for training
            
        Returns:
            Dictionary with cross-validation results
        """
        # Convert y to indices if one-hot encoded
        if y.ndim > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Metrics to track
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y_indices)):
            logger.info(f"Training fold {fold+1}/{n_splits}")
            
            # Split data
            x_train, x_test = x[train_idx], x[test_idx]
            
            # Handle different label formats
            if y.ndim > 1:
                y_train, y_test = y[train_idx], y[test_idx]
            else:
                y_train = to_categorical(y[train_idx], num_classes=2)
                y_test = to_categorical(y[test_idx], num_classes=2)
            
            # Build and train model
            model = model_builder()
            
            history = model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                verbose=1
            )
            
            # Evaluate model
            y_score = model.predict(x_test, batch_size=batch_size)
            
            # Create metrics object
            metrics = EvaluationMetrics(y_test, y_score)
            summary = metrics.get_summary()
            
            # Store metrics
            accuracies.append(summary['accuracy'])
            precisions.append(summary['macro_precision'])
            recalls.append(summary['macro_recall'])
            f1_scores.append(summary['macro_f1'])
            aucs.append(summary['macro_auc'])
            
            # Clear model to free memory
            tf.keras.backend.clear_session()
        
        # Calculate overall metrics
        cv_results = {
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'values': [float(acc) for acc in accuracies]
            },
            'precision': {
                'mean': float(np.mean(precisions)),
                'std': float(np.std(precisions)),
                'values': [float(prec) for prec in precisions]
            },
            'recall': {
                'mean': float(np.mean(recalls)),
                'std': float(np.std(recalls)),
                'values': [float(rec) for rec in recalls]
            },
            'f1': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'values': [float(f1) for f1 in f1_scores]
            },
            'auc': {
                'mean': float(np.mean(aucs)),
                'std': float(np.std(aucs)),
                'values': [float(auc_val) for auc_val in aucs]
            }
        }
        
        # Print results
        logger.info(f"Cross-validation results for {model_name}:")
        logger.info(f"Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        logger.info(f"Precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
        logger.info(f"Recall: {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")
        logger.info(f"F1 Score: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
        logger.info(f"AUC: {cv_results['auc']['mean']:.4f} ± {cv_results['auc']['std']:.4f}")
        
        # Plot results
        self._plot_cv_results(cv_results, model_name)
        
        # Save results to JSON
        cv_path = os.path.join(self.figures_dir, f'{model_name}_cv_results.json')
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=4)
        logger.info(f"Cross-validation results saved to {cv_path}")
        
        return cv_results
    
    def _plot_cv_results(self, cv_results: Dict[str, Dict[str, Any]], 
                        model_name: str) -> plt.Figure:
        """
        Plot cross-validation results.
        
        Args:
            cv_results: Cross-validation results
            model_name: Name of the model
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Metrics to plot
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        x_pos = np.arange(len(metrics))
        
        # Extract means and stds
        means = [cv_results[metric]['mean'] for metric in metrics]
        stds = [cv_results[metric]['std'] for metric in metrics]
        
        # Plot bars
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)
        
        # Set labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.set_title(f'{model_name} Cross-Validation Results')
        
        # Add values on top of bars
        for i, v in enumerate(means):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.figures_dir, f'{model_name}_cv_results.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cross-validation plot saved to {fig_path}")
        
        # Show the figure if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def compare_models(self, models: Dict[str, tf.keras.Model], 
                      x_test: np.ndarray, y_test: np.ndarray,
                      batch_size: int = 32) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same test data.
        
        Args:
            models: Dictionary of model name to model
            x_test: Test data
            y_test: Test labels
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary of model name to evaluation metrics
        """
        # Store original model
        original_model = self.model
        
        # Results for each model
        results = {}
        
        # Evaluate each model
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            
            # Set model
            self.model = model
            
            # Evaluate model
            results[name] = self.evaluate_model(x_test, y_test, name, batch_size)
        
        # Restore original model
        self.model = original_model
        
        # Plot comparison
        self._plot_model_comparison(results)
        
        # Save comparison to JSON
        comp_path = os.path.join(self.figures_dir, 'model_comparison.json')
        with open(comp_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Model comparison saved to {comp_path}")
        
        return results
    
    def _plot_model_comparison(self, results: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            results: Dictionary of model name to evaluation metrics
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Metrics to plot
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Extract model names and values
        model_names = list(results.keys())
        
        # Plot each metric
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            values = [results[name][metric] for name in model_names]
            
            # Plot bars
            axes[i].bar(model_names, values, alpha=0.7)
            
            # Set labels and title
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim([0, 1])
            axes[i].set_title(title)
            
            # Add values on top of bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center')
            
            # Rotate x-tick labels if needed
            if len(model_names) > 3:
                axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.figures_dir, 'model_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {fig_path}")
        
        # Show the figure if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def set_model(self, model) -> None:
        """
        Set the model to evaluate.
        
        Args:
            model: TensorFlow model
        """
        self.model = model
    
    def set_show_plots(self, show_plots: bool) -> None:
        """
        Set whether to show plots.
        
        Args:
            show_plots: Whether to show plots
        """
        self.show_plots = show_plots


# Legacy Evaluator class for backwards compatibility
class LegacyEvaluator:
    """Legacy Evaluator class for backwards compatibility."""
    
    def __init__(self, model, show=True):
        """
        Initialize the legacy evaluator.
        
        Args:
            model: Model to evaluate
            show: Whether to show plots
        """
        self.model = model
        self.show = show
        
        # Create modern evaluator
        self.evaluator = Evaluator(model, show_plots=show)
    
    def plot_accloss_graph(self, history, name):
        """
        Plot accuracy and loss graph.
        
        Args:
            history: Training history
            name: Name of the model
        """
        self.evaluator.plot_training_history(history, name)
    
    def plot_cm(self, y_true, y_pred, name):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            name: Name of the model
        """
        self.evaluator.plot_confusion_matrix(y_true, y_pred, name)
    
    def plot_roc(self, y_true, y_score, name):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_score: Prediction probabilities
            name: Name of the model
        """
        self.evaluator.plot_roc_curve(y_true, y_score, name)
    
    def predict_test_data(self, x, y, name):
        """
        Predict test data and evaluate.
        
        Args:
            x: Test data
            y: Test labels
            name: Name of the model
        """
        # Start timer
        start = time.time()
        
        # Get predictions
        predictions = self.model.predict(x)
        
        # End timer
        end = time.time()
        
        print(f'{name} completed the predictions in {round((end - start), 2)}s '
             f'({round(len(predictions)/(end - start), 2)}ps)')
        
        # Convert one-hot to indices if needed
        if y.ndim > 1:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y
        
        if predictions.ndim > 1:
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_pred = predictions
        
        # Count correct predictions
        count = np.sum(y_true == y_pred)
        accuracy = count / len(predictions)
        
        # Plot confusion matrix
        self.plot_cm(y_true, y_pred, name)
        
        # Plot ROC curve
        self.plot_roc(y, predictions, name)
        
        print(f'Accuracy: {round((accuracy*100), 2)}%')
    
    def set_model(self, model):
        """
        Set the model to evaluate.
        
        Args:
            model: TensorFlow model
        """
        self.model = model
        self.evaluator.set_model(model)
    
    def set_show(self, show):
        """
        Set whether to show plots.
        
        Args:
            show: Whether to show plots
        """
        self.show = show
        self.evaluator.set_show_plots(show)
    
    def get_model(self):
        """
        Get the model being evaluated.
        
        Returns:
            TensorFlow model
        """
        return self.model


# For backwards compatibility
Evaluator.plot_accloss_graph = lambda self, history, name: self.plot_training_history(history, name)
Evaluator.plot_cm = lambda self, y_true, y_pred, name: self.plot_confusion_matrix(y_true, y_pred, name)
Evaluator.plot_roc = lambda self, y_true, y_score, name: self.plot_roc_curve(y_true, y_score, name)
Evaluator.predict_test_data = lambda self, x, y, name: self.evaluate_model(x, y, name)
Evaluator.set_show = lambda self, show: self.set_show_plots(show)
Evaluator.get_model = lambda self: self.model


# If used directly
if __name__ == "__main__":
    import argparse
    import tensorflow as tf
    
    parser = argparse.ArgumentParser(description='Thirdeye evaluation module')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the model file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the test data')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to the test labels')
    parser.add_argument('--name', type=str, default='model',
                       help='Name of the model')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show plots')
    
    args = parser.parse_args()
    
    # Check if files exist
    for path in [args.model, args.data, args.labels]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
    
    # Load model
    try:
        model = tf.keras.models.load_model(args.model)
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load data
    try:
        x_test = np.load(args.data)
        y_test = np.load(args.labels)
        print(f"Data loaded: {x_test.shape}, Labels loaded: {y_test.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = Evaluator(model, show_plots=not args.no_show)
    
    # Evaluate model
    evaluator.evaluate_model(x_test, y_test, args.name)
    
    # Analyze failures
    evaluator.analyze_failures(x_test, y_test, args.name)
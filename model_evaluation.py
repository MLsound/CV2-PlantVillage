# eval_utils.py
import time
import os
import psutil
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def count_keras_trainable_parameters(model: keras.Model) -> int:
    """
    Counts the number of trainable parameters in a Keras model.

    Args:
        model (keras.Model): The Keras model.

    Returns:
        int: The number of trainable parameters.
    """
    return model.count_params()

from typing import Dict, Callable, Optional, List
import tensorflow as tf
from tensorflow import keras
import numpy as np

def evaluate_multiple_models(
    models: Dict[str, keras.Model],
    dataloader: tf.data.Dataset,
    compiled_metrics_names: List[str],
    metrics: Dict[str, Callable],
    target_transform: Optional[Callable] = None,
    output_transform: Optional[Callable] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a series of Keras models on a given dataloader using specified metrics.

    Args:
        models (Dict[str, keras.Model]): A dictionary where keys are model names
            and values are the Keras models.
        dataloader (tf.data.Dataset): The TensorFlow data loader to use for evaluation.
        metrics (Dict[str, Callable]): A dictionary of metric names and
            corresponding callable functions (e.g., {'accuracy': accuracy_score}).
        target_transform (Optional[Callable]): A function to transform the target
            variable (y_true).
        output_transform (Optional[Callable]): A function to transform the model
            output before converting it to predictions.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are model names and
            values are dictionaries of metric names and their calculated values
            for that model.
    """
    all_results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        model_results = {}
        eval_results = model.evaluate(dataloader, verbose=0)
        model_metrics_names = model.metrics_names
        print("Model metrics names:", model_metrics_names)
        if len(compiled_metrics_names) != len(eval_results):
            print(f"Warning: Number of compiled metric names ({len(compiled_metrics_names)}) "
                  f"does not match the number of evaluation results ({len(eval_results)}). "
                  "Results might be misaligned.")

        for i, metric_name in enumerate(compiled_metrics_names):
            if i < len(eval_results):
                model_results[metric_name] = eval_results[i + 1]
            else:
                model_results[metric_name] = float('nan')
                print(f"Warning: Metric '{metric_name}' not found in evaluation results.")

        for metric_name, metric_func in metrics.items():
            print(f"compiled metrics: '{compiled_metrics_names}'...")
            if metric_name not in compiled_metrics_names:
                print(f"Calculating metric '{metric_name}'...")
                all_targets = []
                all_predictions = []
                for batch in dataloader:
                    inputs, targets = batch
                    predictions = model.predict(inputs, verbose=0)
                    if output_transform:
                        predictions = output_transform(predictions)
                    all_targets.extend(targets.numpy())
                    all_predictions.extend(predictions)
                try:
                    targets_np = np.array(all_targets)
                    predictions_np = np.array(all_predictions)
                    if target_transform:
                        targets_np = target_transform(targets_np)
                    model_results[metric_name] = metric_func(targets_np, predictions_np)
                except Exception as e:
                    print(f"Error calculating metric '{metric_name}': {e}")
                    model_results[metric_name] = float('nan')
        all_results[model_name] = model_results
    return all_results


def plot_model_comparison(df: pd.DataFrame,
                           metrics_to_plot: List[str]
                           ) -> None:
    """
    Plots the comparison of models based on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the comparison results
            (output of compare_models).
        metrics_to_plot (List[str]): A list of metrics to plot
            (e.g., ['size_mb', 'trainable_params', 'val_accuracy']).
        dataloader_names (List[str]): A list of dataloader names to include
            in the plot labels (e.g., ['train', 'val', 'test']).
    """
    num_models = len(df)
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))
    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable even if only one subplot

    x = np.arange(num_models)
    width = 0.8  # the width of the bars
    model_names = df['model_name']

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        values = df[metric]
        ax.bar(x, values, width)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim(0, max(values) * 1.1) #Adjust y-axis

    plt.tight_layout()
    plt.show()

def compare_training_times(models: Dict[str, dict]) -> pd.DataFrame:
    """
    Compares the training times of different models.  Assumes the input
    'models' dictionary contains a 'training_time' key in the stats.

    Args:
        models (Dict[str, dict]): A dictionary of model names and
            dictionaries, where each dictionary contains a 'training_time' key.
            For example:
            {
                'ModelA': {'training_time': 120.5, 'other_stat': 0.9},
                'ModelB': {'training_time': 150.2, 'other_stat': 0.8},
            }

    Returns:
        pd.DataFrame: A DataFrame containing the model names and training times.
    """
    training_times = []
    for model_name, model_stats in models.items():
        if 'training_time' in model_stats:
            training_times.append({'model_name': model_name, 'training_time': model_stats['training_time']})
        else:
            print(f"Warning: Model '{model_name}' does not have 'training_time' in its stats.")
            training_times.append({'model_name': model_name, 'training_time': float('nan')})
    return pd.DataFrame(training_times)

def plot_training_times(df: pd.DataFrame) -> None:
    """
    Plots the training times of different models from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing model names and training times
            (output of compare_training_times).
    """
    plt.figure(figsize=(8, 5))
    plt.bar(df['model_name'], df['training_time'])
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


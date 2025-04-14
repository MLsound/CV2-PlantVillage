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

def get_keras_model_size(model: keras.Model, unit: str = 'mb') -> float:
    """
    Calculates the size of a Keras model in memory.

    Args:
        model (keras.Model): The Keras model.
        unit (str): The unit of measurement ('mb', 'kb', or 'bytes').

    Returns:
        float: The size of the model in the specified unit.
    """
    # We will save the model to a temporary file and get its size
    temp_file = "temp_model.h5"
    model.save(temp_file)
    size_bytes = os.path.getsize(temp_file)
    os.remove(temp_file)  # Clean up the temporary file
    print(f"Model size: {size_bytes} bytes")
    if unit.lower() == 'mb':
        return size_bytes / (1024 * 1024)
    elif unit.lower() == 'kb':
        return size_bytes / 1024
    elif unit.lower() == 'bytes':
        return size_bytes
    else:
        raise ValueError("Invalid unit. Must be 'mb', 'kb', or 'bytes'")

def count_keras_trainable_parameters(model: keras.Model) -> int:
    """
    Counts the number of trainable parameters in a Keras model.

    Args:
        model (keras.Model): The Keras model.

    Returns:
        int: The number of trainable parameters.
    """
    return model.count_params()

def calculate_keras_metrics(
    model: keras.Model,
    dataloader: tf.data.Dataset,
    metrics: Dict[str, Callable],  # Change back to Dict[str, Callable]
    target_transform: Optional[Callable] = None,
    output_transform: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Calculates specified metrics for a given Keras model on a dataloader.

    Args:
        model (keras.Model): The Keras model.
        dataloader (tf.data.Dataset): The TensorFlow data loader.
        metrics (Dict[str, Callable]): A dictionary of metric names and
            corresponding callable functions (e.g., {'accuracy': accuracy_score}).
            These should be metrics that the Keras model was compiled with.
        target_transform (Optional[Callable]): A function to transform the target
            variable (y_true).
        output_transform (Optional[Callable]): A function to transform the model
            output before converting it to predictions.

    Returns:
        Dict[str, float]: A dictionary of metric names and their calculated values.
    """
    results = {}
    eval_results = model.evaluate(dataloader, verbose=0)

    # Get the names of the metrics the model was compiled with.
    model_metrics_names = model.metrics_names

    for metric_name, metric_func in metrics.items():
        if metric_name in model_metrics_names:
            # If the metric is already calculated by the model, use that value.
            metric_index = model_metrics_names.index(metric_name)
            results[metric_name] = eval_results[metric_index]
        else:
            # Otherwise, calculate the metric using the provided function.
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
                results[metric_name] = metric_func(targets_np, predictions_np)
            except Exception as e:
                print(f"Error calculating metric {metric_name}: {e}")
                results[metric_name] = float('nan')
    return results
def compare_keras_models(
    models: Dict[str, keras.Model],
    dataloaders: Dict[str, tf.data.Dataset],
    metrics: Dict[str, Callable],  # Change back to Dict[str, Callable]
    target_transform: Optional[Callable] = None,
    output_transform: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Compares the performance of multiple Keras models.

    Args:
        models (Dict[str, keras.Model]): A dictionary of model names and
            corresponding Keras models.
        dataloaders (Dict[str, tf.data.Dataset]): A dictionary of
            dataloader names ('train', 'val', 'test') and their corresponding
            TensorFlow data loaders.
        metrics (Dict[str, Callable]): A dictionary of metric names and
            corresponding callable functions (e.g., {'accuracy': accuracy_score}).
        target_transform (Optional[Callable]): A function to transform the target.
        output_transform (Optional[Callable]): A function to transform the output.

    Returns:
        pd.DataFrame: A DataFrame containing the comparison results.
    """
    results = []
    for model_name, model in models.items():
        model_results = {
            'model_name': model_name,
            'size_mb': 0,
            'trainable_params': count_keras_trainable_parameters(model),
        }

        # Time the evaluation
        start_time = time.time()
        for dataloader_name, dataloader in dataloaders.items():
            model_metrics = calculate_keras_metrics(model, dataloader, metrics,
                                                  target_transform, output_transform)
            for metric_name, metric_value in model_metrics.items():
                model_results[f'{dataloader_name}_{metric_name}'] = metric_value
        end_time = time.time()
        model_results['evaluation_time'] = end_time - start_time
        results.append(model_results)

    return pd.DataFrame(results)

def plot_model_comparison(df: pd.DataFrame,
                           metrics_to_plot: List[str],
                           dataloader_names: List[str]) -> None:
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

def compare_and_plot_models(
    models: Dict[str, keras.Model],
    model_stats: Dict[str, dict],
    dataloaders: Dict[str, tf.data.Dataset],
    metrics: Dict[str, Callable],  # Change back to Dict[str, Callable]
    target_transform: Optional[Callable] = None,
    output_transform: Optional[Callable] = None,
    dataloader_names: List[str] =  ['train', 'val', 'test']
) -> None:
    """
    Combines model comparison and plotting into a single function.

    Args:
        models (Dict[str, keras.Model]): A dictionary of model names and
            corresponding Keras models.
        model_stats (Dict[str, dict]): A dictionary of model name and model stats
        dataloaders (Dict[str, tf.data.Dataset]): A dictionary of
            dataloader names and their corresponding TensorFlow data loaders.
        metrics (Dict[str, Callable]): A dictionary of metric names and
            corresponding callable functions.
        target_transform (Optional[Callable]): Target transform function.
        output_transform (Optional[Callable]): Output transform function.
        dataloader_names (List[str]): dataloader names
    """
    comparison_df = compare_keras_models(models, dataloaders, metrics, target_transform, output_transform)
    print("Model Comparison:")
    print(comparison_df)

    metrics_to_plot = ['size_mb', 'trainable_params'] + [f'{dl_name}_{metric_name}' for dl_name in dataloader_names for metric_name in metrics.keys()]
    plot_model_comparison(comparison_df, metrics_to_plot, dataloader_names)

    training_times_df = compare_training_times(model_stats)
    print("\nTraining Time Comparison:")
    print(training_times_df)
    plot_training_times(training_times_df)

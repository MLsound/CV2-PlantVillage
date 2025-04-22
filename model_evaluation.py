# eval_utils.py
# Librerías estándar de Python
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

# Librerías de terceros
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Métricas de ML
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import register_keras_serializable

def count_keras_trainable_parameters(model: keras.Model) -> int:
    """
    Counts the number of trainable parameters in a Keras model.

    Args:
        model (keras.Model): The Keras model.

    Returns:
        int: The number of trainable parameters.
    """
    return model.count_params()


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


def plot_model_comparison(df: pd.DataFrame, metrics_to_plot: List[str]) -> None:
    """
    Plots the comparison of models based on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the comparison results
            (output of compare_models).
        metrics_to_plot (List[str]): A list of metrics to plot
            (e.g., ['size_mb', 'trainable_params', 'val_accuracy']).
    """
    num_models = len(df)
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable even if only one subplot

    model_names = df.index

    # Use Seaborn color palette for the bars
    palette = sns.color_palette("Set2", n_colors=num_models)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        values = df[metric]

        # Create the bar plot using Seaborn
        sns.barplot(x=model_names, y=values, ax=ax, palette=palette);

        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} ')
        ax.set_ylim(min(values)*0.9, max(values) * 1.1)  # Adjust y-axis

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()
    return


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

@register_keras_serializable()
def macro_f1_score(y_true, y_pred):
    y_true_labels = tf.argmax(y_true, axis=-1, output_type=tf.int64)
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int64)

    num_classes = tf.shape(y_true)[-1]

    def compute_f1(i):
        i = tf.cast(i, tf.int64)  # 👈 match dtype with y_true_labels
        y_true_i = tf.cast(tf.equal(y_true_labels, i), tf.float32)
        y_pred_i = tf.cast(tf.equal(y_pred_labels, i), tf.float32)

        tp = tf.reduce_sum(y_true_i * y_pred_i)
        fp = tf.reduce_sum((1 - y_true_i) * y_pred_i)
        fn = tf.reduce_sum(y_true_i * (1 - y_pred_i))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    f1s = tf.map_fn(compute_f1, tf.range(num_classes), dtype=tf.float32)
    return tf.reduce_mean(f1s)



def recalculate_metrics(model_name, model, dataloader, metrics, output_transform=None, target_transform=None):

    model_results = {}
    all_targets, all_predictions = [], []

    # Run predictions over the dataloader
    for batch in dataloader:
        inputs, targets = batch
        predictions = model.predict(inputs, verbose=0)
        if output_transform:
            predictions = output_transform(predictions)
        all_targets.extend(targets.numpy())
        all_predictions.extend(predictions)

    targets_np = np.array(all_targets)
    predictions_np = np.array(all_predictions)

    if target_transform:
        targets_np = target_transform(targets_np)

    # Calculate each provided metric
    for metric_name, metric_func in metrics.items():
        try:
            value = metric_func(targets_np, predictions_np)
            model_results[metric_name] = np.mean(value) if isinstance(value, (list, np.ndarray)) else value
        except Exception as e:
            print(f"Error calculating metric '{metric_name}': {e}")
            model_results[metric_name] = float('nan')

    return {model_name: model_results}
def recalculate_metrics2(model_name, model, dataloader, metrics, output_transform=None, target_transform=None):
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    from collections import Counter

    model_results = {}
    all_targets, all_predictions = [], []

    # Run predictions over the dataloader
    for batch in dataloader:
        inputs, targets = batch
        predictions = model.predict(inputs, verbose=0)
        if output_transform:
            predictions = output_transform(predictions)
        all_targets.extend(targets.numpy())
        all_predictions.extend(predictions)

    targets_np = np.array(all_targets)
    predictions_np = np.array(all_predictions)

    if target_transform:
        targets_np = target_transform(targets_np)

    # Get class support counts
    target_labels = targets_np.flatten()
    class_counts = Counter(target_labels)
    classes = sorted(class_counts.keys())
    supports = np.array([class_counts[c] for c in classes])
    total_support = np.sum(supports)

    # Calculate each provided metric
    for metric_name, metric_func in metrics.items():
        try:
            value = metric_func(targets_np, predictions_np)
            if isinstance(value, (list, np.ndarray)) and len(value) == len(supports):
                # Weighted average
                weighted_value = np.sum(np.array(value) * supports) / total_support
                model_results[metric_name] = weighted_value
            else:
                # Scalar metric (like accuracy)
                model_results[metric_name] = value
        except Exception as e:
            print(f"Error calculating metric '{metric_name}': {e}")
            model_results[metric_name] = float('nan')

    return {model_name: model_results}



def plot_metric_histories(histories: dict, metrics: list = ['accuracy', 'f1_score']):
    """
    Plots training and validation loss separately from other specified metrics for each model.
    Supports multi-dimensional metrics (e.g., per-class f1_score) by taking the mean.
    """
    for model_name, history in histories.items():
        if history is None:
            print(f"❌ No history found for {model_name}")
            continue

        print(f"\n📊 Plotting metrics for {model_name}...")

        # --- 1. Plot LOSS ---
        if 'loss' in history:
            plt.figure(figsize=(8, 5))
            plt.title(f'Loss History for {model_name}')
            loss = [np.mean(v) if hasattr(v, '__iter__') else v for v in history['loss']]
            plt.plot(loss, label='Training Loss', linewidth=2)

            if 'val_loss' in history:
                val_loss = [np.mean(v) if hasattr(v, '__iter__') else v for v in history['val_loss']]
                plt.plot(val_loss, label='Validation Loss', linewidth=2, linestyle='--')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ No 'loss' found in history for {model_name}")

        # --- 2. Plot other metrics (excluding loss) ---
        other_metrics = [m for m in metrics if m not in ['loss']] # Exclude 'loss'
        for metric in other_metrics:
            has_data = False
            plt.figure(figsize=(8, 5))
            plt.title(f'{metric.capitalize()} History for {model_name}')

            # Training metric
            if metric in history:
                values = history[metric]
                reduced = [np.mean(v) if hasattr(v, '__iter__') else v for v in values]
                plt.plot(reduced, label=f'Training {metric}', linewidth=2)
                has_data = True

            # Validation metric
            val_metric = f'val_{metric}'
            if val_metric in history:
                val_values = history[val_metric]
                reduced_val = [np.mean(v) if hasattr(v, '__iter__') else v for v in val_values]
                plt.plot(reduced_val, label=f'Validation {metric}', linewidth=2, linestyle='--')
                has_data = True

            if has_data:
                plt.xlabel('Epochs')
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                plt.close()
                print(f"⚠️ No data for metric '{metric}' in {model_name}")
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix_from_df(df: pd.DataFrame, model_name: str, class_names=None, normalize=False):
    """
    Plots a confusion matrix for a specific model from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the confusion matrix.
        model_name (str): The model name (index of the DataFrame).
        class_names (List[str], optional): Class labels for axes.
        normalize (bool): Whether to normalize the matrix row-wise.
    """
    try:
        cm = df.loc[model_name, 'confusion_matrix']
    except KeyError:
        print(f"Model '{model_name}' not found in DataFrame.")
        return

    if normalize:
        cm = cm.astype(np.float32)
        cm /= cm.sum(axis=1, keepdims=True)

    num_classes = cm.shape[0]
    figsize = max(12, int(num_classes * 0.5))  # scale size with number of classes

    plt.figure(figsize=(figsize, figsize))
    sns.heatmap(
        cm,
        annot=False,  # Set to True if you want to see individual cell values
        fmt=".2f" if normalize else "d",
        cmap="cividis",
        xticklabels=class_names if class_names else np.arange(num_classes),
        yticklabels=class_names if class_names else np.arange(num_classes),
        cbar_kws={'shrink': 0.5},
        square=True
    )

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(f"Confusion Matrix for {model_name}", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.show()

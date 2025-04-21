# eda_utils.py
import json
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_representative_color_histogram(df: pd.DataFrame, load_image_func, root_dir: str = '.', output_csv_file: str = 'representative_histogram.csv'):
    """
    Calculates and saves a representative color histogram of an image dataset.

    Args:
        df (pd.DataFrame): DataFrame containing image information (e.g., file paths).
        load_image_func (callable): Function to load a PIL Image from the DataFrame.
        root_dir (str): Root directory where images are located.
        output_csv_file (str): Name of the CSV file to save the histogram data.

    Returns:
        dict: A dictionary containing the representative red, green, and blue histograms,
              or None if an error occurred or no images were processed.
    """
    representative_histogram = None

    # Try to load existing representative histogram from CSV
    try:
        loaded_df = pd.read_csv(output_csv_file)
        if 'red' in loaded_df.columns and 'green' in loaded_df.columns and 'blue' in loaded_df.columns and len(loaded_df) == 256:
            representative_histogram = {
                'red': loaded_df['red'].tolist(),
                'green': loaded_df['green'].tolist(),
                'blue': loaded_df['blue'].tolist()
            }
            print(f"Loaded existing representative color histogram from {output_csv_file}")
        else:
            print(f"Invalid format in {output_csv_file}. Processing images...")
            representative_histogram = None  # Force processing
    except FileNotFoundError:
        print(f"No existing representative color histogram found at {output_csv_file}. Processing images...")
        representative_histogram = None  # Force processing
    except pd.errors.EmptyDataError:
        print(f"{output_csv_file} is empty. Processing images...")
        representative_histogram = None  # Force processing
    except Exception as e:
        print(f"Error loading histogram from {output_csv_file}: {e}. Processing images...")
        representative_histogram = None  # Force processing

    if representative_histogram is None:
        num_images = len(df)
        all_red_hist = np.zeros(256)
        all_green_hist = np.zeros(256)
        all_blue_hist = np.zeros(256)
        processed_count = 0

        for index in range(num_images):
            try:
                image = load_image_func(data=df, index=index, root=root_dir)
                if image and image.mode == 'RGB':
                    img_array = np.array(image)
                    red_channel = img_array[:, :, 0].flatten()
                    green_channel = img_array[:, :, 1].flatten()
                    blue_channel = img_array[:, :, 2].flatten()

                    hist_r, _ = np.histogram(red_channel, bins=256, range=(0, 256))
                    hist_g, _ = np.histogram(green_channel, bins=256, range=(0, 256))
                    hist_b, _ = np.histogram(blue_channel, bins=256, range=(0, 256))

                    all_red_hist += hist_r
                    all_green_hist += hist_g
                    all_blue_hist += hist_b
                    processed_count += 1
                    image.close()
                    del image, img_array, red_channel, green_channel, blue_channel
                elif image:
                    image.close()
                    del image
            except Exception as e:
                print(f"Error processing image at index {index}: {e}")

        # Normalize the histograms by the number of processed images
        if processed_count > 0:
            representative_histogram = {
                'red': (all_red_hist / processed_count).tolist(),
                'green': (all_green_hist / processed_count).tolist(),
                'blue': (all_blue_hist / processed_count).tolist()
            }

            # Save the representative histogram to csv
            hist_df = pd.DataFrame(representative_histogram)
            hist_df.to_csv(output_csv_file, index=False)
            print(f"Representative color histogram saved to {output_csv_file}")
        else:
            print("No valid images processed to create a representative histogram.")

    return representative_histogram

def plot_representative_color_histogram(representative_histogram: dict):
    """
    Plots the representative color histogram.

    Args:
        representative_histogram (dict): Dictionary containing 'red', 'green', and 'blue'
                                         histogram data.
    """
    if representative_histogram:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.bar(range(256), representative_histogram['red'], color='red', alpha=0.7)
        plt.title('Representative Red Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Average Frequency')

        plt.subplot(1, 3, 2)
        plt.bar(range(256), representative_histogram['green'], color='green', alpha=0.7)
        plt.title('Representative Green Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Average Frequency')

        plt.subplot(1, 3, 3)
        plt.bar(range(256), representative_histogram['blue'], color='blue', alpha=0.7)
        plt.title('Representative Blue Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Average Frequency')

        plt.tight_layout()
        plt.show()
    else:
        print("No representative color histogram available for plotting.")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def plot_distribution(df, field, plot_type='bar', top_n=None, filter_by=None, ax=None, **kwargs):
    """
    Plots the distribution of values in a given field from a DataFrame on a given Axes object.

    Parameters:
    - df: pandas DataFrame
    - field: column name to group by (e.g., 'class', 'group', 'tag')
    - plot_type: 'bar' for bar chart, 'pie' for pie chart (default: 'bar')
    - top_n: if specified, shows only top N most frequent values
    - filter_by: dict to filter rows before plotting. Example: {'group': 'Tomato'}
    - ax: matplotlib Axes object to plot on. If None, a new figure and axes are created.
    - **kwargs: additional keyword arguments passed to matplotlib or seaborn plotting functions.
               For example, `autopct='%1.1f%%'` for pie charts, `palette` for both.
    """
    filtered_df = df.copy()

    # Apply filtering if specified
    if filter_by:
        for key, value in filter_by.items():
            filtered_df = filtered_df[filtered_df[key] == value]

    # Get value counts for the selected field
    value_counts = filtered_df[field].value_counts()

    if top_n:
        value_counts = value_counts.head(top_n)

    title_suffix = f" (filtered: {filter_by})" if filter_by else ""

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == 'bar':
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette=kwargs.get('palette', "viridis"))
        ax.set_title(f'Distribution per {field}{title_suffix}', fontsize=14)
        ax.set_xlabel(field, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right') # Set horizontal alignment here
        ax.tick_params(axis='y', labelsize=12)
    elif plot_type == 'pie':
        ax.pie(value_counts, labels=value_counts.index, startangle=140, **kwargs)
        ax.set_title(f'Distribution of {field}{title_suffix}', fontsize=14)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    else:
        raise ValueError(f"Invalid plot_type: '{plot_type}'. Choose 'bar' or 'pie'.")

    if ax is None:
        plt.tight_layout()
        plt.show()

def plot_density_distribution(df, target):
    df_graph = df.copy()

    # Calculate value counts and order them
    ordered_classes = df_graph[target].value_counts().index

    # Map categorical target values to numeric indices
    df_graph['numeric_target'] = df_graph[target].map({cls: idx for idx, cls in enumerate(ordered_classes)})

    plt.figure(figsize=(15, 10))

    sns.histplot(data=df_graph, x='numeric_target', hue='numeric_target', discrete=True, stat='density', label='Histogram', palette="viridis", legend=False)
    sns.kdeplot(data=df_graph, x='numeric_target', label='Density Plot', color='orange', linewidth=2)
    plt.xticks(ticks=range(len(ordered_classes)), labels=ordered_classes, rotation=45, ha='right', va='top')
    plt.title(f"Histogram and Density plot of '{target}' distribution")
    plt.xlabel("")
    plt.ylabel("Density")
    plt.legend(labels=["Histogram", "Density Plot"])

    plt.tight_layout()
    plt.show()

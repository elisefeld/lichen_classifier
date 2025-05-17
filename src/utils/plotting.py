import math
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
cfg = Config()

# Set random seeds
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)

plt.style.use('seaborn-v0_8-colorblind')
sns.set_theme(context='talk', style='white')

### Functions ###
def plot_class_distribution(df: pd.DataFrame, filter: bool = cfg.filter_classes):
    if filter:
        df = df[df['genus'].isin(cfg.train_classes)].copy()
    class_counts = df['genus'].value_counts()
    plt.figure(figsize=(12, 10))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Taxonomic Genus')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(cfg.EDA_dir/'plot_class_distribution.png', bbox_inches='tight')
    plt.close()


def plot_time(df: pd.DataFrame, column: str = 'observed_on_month', type: str = 'Month', filter: bool = True):
    if filter:
        df = df[df['genus'].isin(cfg.train_classes)].copy()
    if column == 'observed_on_year':
        df = df[df['observed_on_year'] > 2008].copy()
    plt.figure(figsize=(12, 6))
    df[column].value_counts().sort_index().plot(kind='line')
    plt.title(f'Number of Observations by {type}')
    plt.ylabel('Count')
    plt.xlabel(type)
    plt.savefig(cfg.EDA_dir/f'plot_{type}.png', bbox_inches='tight')
    plt.close()


def plot_location(df: pd.DataFrame, filter: bool = cfg.filter_classes, facet_type: str = 'facetted'):
    if filter:
        df = df[df['genus'].isin(cfg.train_classes)].copy()

    if facet_type == 'facetted':
        g = sns.FacetGrid(df, col='genus', col_wrap=5,
                          height=3, hue='morphology')
        g.map(sns.scatterplot, 'longitude', 'latitude', alpha=0.5, s=10)
        g.figure.subplots_adjust(top=0.9)
        g.set_axis_labels('Longitude', 'Latitude')
        g.figure.suptitle('Geospatial Distribution of Observations')

    else:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='longitude', y='latitude',
                        hue='morphology', alpha=0.5, s=10)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geospatial Distribution of Observations by Morphology')

    plt.savefig(
        cfg.EDA_dir/f'plot_location_{facet_type}.png', bbox_inches='tight')
    plt.close()


def plot_rgb_histograms(df: tf.data.Dataset, class_names: list, bins: int = 256, cols: int = 4):
    histograms = {
        class_name: {'r': np.zeros(bins), 'g': np.zeros(
            bins), 'b': np.zeros(bins), 'count': 0}
        for class_name in class_names
    }

    for batch_images, batch_labels in df:
        for img, label in zip(batch_images, batch_labels):
            label = tf.argmax(label, axis=-1).numpy()
            class_name = class_names[label]
            img = img.numpy()
            img = (
                img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

            b = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
            g = cv2.calcHist([img], [1], None, [bins], [0, 256]).flatten()
            r = cv2.calcHist([img], [2], None, [bins], [0, 256]).flatten()

            histograms[class_name]['b'] += b
            histograms[class_name]['g'] += g
            histograms[class_name]['r'] += r
            histograms[class_name]['count'] += 1

    for class_name in class_names:
        count = histograms[class_name]['count']
        if count > 0:
            for ch in ['r', 'g', 'b']:
                h = histograms[class_name][ch]
                histograms[class_name][ch] = h / h.sum()

    n = len(class_names)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(
        cols * 5, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    x = np.arange(bins)
    for i, class_name in enumerate(class_names):
        ax = axes[i]
        r = histograms[class_name]['r']
        g = histograms[class_name]['g']
        b = histograms[class_name]['b']
        ax.plot(x, r, color='red', label='Red')
        ax.plot(x, g, color='green', label='Green')
        ax.plot(x, b, color='blue', label='Blue')
        ax.set_title(class_name)
        ax.set_xlim(0, bins - 1)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Add one legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    fig.suptitle('RGB Channel Histograms per Genus')
    plt.savefig(cfg.EDA_dir/'plot_rgb_histograms.png', bbox_inches='tight')
    plt.close()

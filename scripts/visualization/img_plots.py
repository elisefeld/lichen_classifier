import math
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def color_histogram(df, class_names, bins: int = 256):
    avg_histograms = {
        class_name: {
            'r': np.zeros(bins),
            'g': np.zeros(bins),
            'b': np.zeros(bins),
            'count': 0
        } for class_name in class_names
    }

    for batch_images, batch_labels in df:
        for img, label in zip(batch_images, batch_labels):
            label = tf.argmax(label, axis=-1).numpy()
            class_name = class_names[label]
            img = img.numpy()
            img = (
                img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

            b_hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
            g_hist = cv2.calcHist([img], [1], None, [bins], [0, 256]).flatten()
            r_hist = cv2.calcHist([img], [2], None, [bins], [0, 256]).flatten()

            avg_histograms[class_name]['b'] += b_hist
            avg_histograms[class_name]['g'] += g_hist
            avg_histograms[class_name]['r'] += r_hist
            avg_histograms[class_name]['count'] += 1

    for class_name in class_names:
        count = avg_histograms[class_name]['count']
        if count == 0:
            continue
        r = avg_histograms[class_name]['r'] / count
        g = avg_histograms[class_name]['g'] / count
        b = avg_histograms[class_name]['b'] / count
        x = np.arange(bins)

        plt.figure(figsize=(10, 6))
        plt.plot(x, r, color='red', label='Red Channel')
        plt.plot(x, g, color='green', label='Green Channel')
        plt.plot(x, b, color='blue', label='Blue Channel')
        plt.title(f'Average Color Channel Histogram for {class_name}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()


def rgb_histograms_grid(df, class_names, bins=256, cols=4):
    # Collect histograms
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
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

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
        cols * 4, rows * 3), sharex=True, sharey=True)
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

    fig.suptitle('RGB Channel Histograms per Genus', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_class_distribution(df: pd.DataFrame, class_names: list):
    class_counts = df['genus'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Genus')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

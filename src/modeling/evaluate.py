from config import Config
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import seaborn as sns
import json

plt.style.use('seaborn-v0_8-colorblind')
sns.set_theme(context='talk', style='white')

cfg = Config()

# Set random seeds
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)

### Functions ###


def train_and_evaluate(model: keras.Model,
                       train_ds: tf.data.Dataset,
                       val_ds: tf.data.Dataset,
                       test_ds: tf.data.Dataset,
                       optimizer: keras.optimizers.Optimizer,
                       callbacks: list,
                       class_weights: np.ndarray,
                       test_type: str,
                       trial: int):
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.smoothing),
        optimizer=optimizer,
        metrics=[*cfg.metrics]
    )

    # Print model summary
    model.summary()

    visualize_model(model, test_type=test_type) # FIX for fine tuning base model

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    results = model.evaluate(test_ds, return_dict=True)

    path = cfg.get_file_name(dir=cfg.training_history_dir,
                             file_type='results', ext='json', test_type=test_type)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    evaluator = ModelEvaluator(
        model, test_ds, trial=trial, test_type=test_type)

    evaluator.save_history(history)
    evaluator.save_class_metrics()
    evaluator.plot_history(history)
    evaluator.plot_confusion_matrix()
    evaluator.visualize_predictions()
    return history


class ModelEvaluator:
    def __init__(self, model, test_ds, test_type: str = None, trial: int = None, results_dir: Path = cfg.results_dir):
        self.model = model
        self.test_ds = test_ds
        self.test_classes = cfg.test_classes
        self.results_dir = results_dir
        self.y_true = None
        self.y_pred = None
        self.trial = trial
        self.test_type = test_type

    def get_true_pred_vals(self):
        if self.y_true is None or self.y_pred is None:
            predictions = self.model.predict(self.test_ds)
            self.y_pred = np.argmax(predictions, axis=1)
            self.y_true = np.concatenate([np.argmax(labels.numpy(), axis=1)
                                          for _, labels in self.test_ds])
        return self.y_true, self.y_pred

    def save_history(self, history):
        path = cfg.get_file_name(
            cfg.training_history_dir, 'training_history', 'json', test_type=self.test_type)
        history_dict = history.history
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=4)

    def save_class_metrics(self):
        path = cfg.get_file_name(
            cfg.class_metrics_dir, 'class_metrics', 'csv', test_type=self.test_type)
        y_true, y_pred = self.get_true_pred_vals()
        report = classification_report(
            y_true, y_pred, target_names=self.test_classes, output_dict=True)
        print("Classification Report:", report)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(path, sep='\t')

    def plot_history(self, history): # FIX THIS!!!!!!!
        path = cfg.get_file_name(
            cfg.training_history_dir, 'training_history', 'png', test_type=self.test_type)
        
        metrics = list(cfg.metrics) + ['loss']
        val_metrics = [f'val_{metric}' for metric in metrics]

        print(f"Available keys in history: {list(history.history.keys())}")

        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

        for i, (metric, val_metric) in enumerate(zip(metrics, val_metrics)):
            ax = axes[i] if num_metrics > 1 else axes  # Handle single subplot case

            if metric not in history.history or val_metric not in history.history:
                print(f"Metric '{metric}' or '{val_metric}' not found in history.")
                ax.set_title(f"Metric '{metric}' not found")
                continue

            ax.plot(history.history[metric], label=f'Train {metric}')
            ax.plot(history.history[val_metric], label=f'Validation {metric}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.set_title(f'Training and Validation {metric.capitalize()}')
            
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self):
        path = cfg.get_file_name(
            cfg.confusion_matrix_dir, 'confusion_matrix', 'png', test_type=self.test_type)

        y_true, y_pred = self.get_true_pred_vals()

        cm = confusion_matrix(
            y_true, y_pred, labels=range(len(self.test_classes)))
        plt.figure(figsize=(10, 10))

        sns.heatmap(
            cm,
            annot=True,
            xticklabels=self.test_classes,
            yticklabels=self.test_classes,
            annot_kws={"size": 8, "color": "black"},
            linewidths=0.5,
            cbar_kws={"label": "Count"}
        )
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def visualize_predictions(self, num_images=5):
        path = cfg.get_file_name(
            cfg.confusion_matrix_dir, 'predictions', 'png', test_type=self.test_type)

        for batch_images, batch_labels in self.test_ds.take(1):
            probs = self.model.predict(batch_images)
            top3_preds = tf.math.top_k(probs, k=3)
            top3_indices = top3_preds.indices.numpy()
            top3_values = top3_preds.values.numpy()
            true_labels = tf.argmax(batch_labels, axis=1).numpy()

            plt.figure(figsize=(15, num_images * 3))

            for i in range(num_images):
                img = batch_images[i].numpy()
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                true_label = self.test_classes[true_labels[i]]
                pred_labels = [self.test_classes[j] for j in top3_indices[i]]
                confidences = top3_values[i]

                # Compose title
                correct = (top3_indices[i][0] == true_labels[i])
                title_color = 'green' if correct else 'red'
                title = f"True: {true_label}\n"
                for rank in range(3):
                    label = pred_labels[rank]
                    conf = confidences[rank]
                    title += f"{rank+1}. {label} ({conf:.2%})\n"

                # Plot
                ax = plt.subplot(num_images, 1, i + 1)
                plt.imshow(img)
                plt.axis("off")
                ax.set_title(title, color=title_color, fontsize=10, loc='left')

            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            break  # Only one batch


def visualize_model(model, test_type: str = None):
    keras.utils.plot_model(model,
                           to_file=f'model{test_type}.png',
                           show_shapes=True,
                           show_dtype=False,
                           show_layer_names=True,
                           rankdir="TB",
                           expand_nested=False,
                           dpi=200,
                           show_layer_activations=True,
                           show_trainable=True,
                           )

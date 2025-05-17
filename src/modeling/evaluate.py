from config import Config
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import json
import logging


plt.style.use('seaborn-v0_8-colorblind')
sns.set_theme(context='talk', style='white')

cfg = Config()

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)

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

    if test_type == 'coarse':
        for layer in model.get_layer("base_model").layers:
            layer.trainable = False
        logger.info("Training with frozen base model...")
    elif test_type == 'fine':
        for layer in model.get_layer("base_model").layers[-cfg.frozen_layers:]:
            layer.trainable = True
        logger.info("Training with unfrozen base model...")

    # Print model summary
    model.summary()

    # FIX for fine tuning base model
    visualize_model(model, test_type=test_type)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    results = model.evaluate(test_ds, return_dict=True)

    logger.info(f"Test results: {results}")

    evaluator = ModelEvaluator(
        model, test_ds, test_type=test_type, trial=trial, history=history, results=results)

    evaluator.save_true_pred_vals()
    evaluator.save_results()
    evaluator.save_history()
    evaluator.save_class_metrics()
    evaluator.visualize_predictions(num_images=5)

def visualize_model(model, test_type: str = None):
    keras.utils.plot_model(model,
                        to_file=f'model{test_type}.png',
                        show_shapes=False,
                        show_dtype=False,
                        show_layer_names=True,
                        rankdir="LR",
                        expand_nested=False,
                        dpi=200,
                        show_layer_activations=True,
                        show_trainable=True,
                        )
        
class ModelEvaluator:
    def __init__(self,
                 model,
                 test_ds,
                 test_type: str = None,
                 trial: int = None,
                 history: dict = None,
                 results: dict = None):
        self.model = model
        self.test_ds = test_ds
        self.test_classes = cfg.test_classes
        self.trial = trial
        self.test_type = test_type
        self.history = history
        self.results = results
        self.y_true = None
        self.y_pred = None

        # Paths
        self.results_dir = cfg.get_file_name(
            dir=cfg.training_data_dir, file_type='results', ext='json', test_type=self.test_type)
        self.history_csv = cfg.get_file_name(
            cfg.training_data_dir, 'training_history', 'csv', test_type=self.test_type)
        self.predictions_csv = cfg.get_file_name(
            cfg.training_data_dir, 'predictions', 'csv', test_type=self.test_type)
        self.metrics_csv = cfg.get_file_name(
            cfg.class_metrics_dir, 'class_metrics', 'csv', test_type=self.test_type)
        self.prediction_imgs_dir = cfg.get_file_name(
            cfg.training_plots_dir, 'predictions', 'png', test_type=self.test_type)

    def save_true_pred_vals(self):
        if self.y_true is None or self.y_pred is None:
            predictions = self.model.predict(self.test_ds)
            self.y_pred = np.argmax(predictions, axis=1)
            self.y_true = np.concatenate([np.argmax(labels.numpy(), axis=1)
                                          for _, labels in self.test_ds])
        true_pred_df = pd.DataFrame({
            'y_true': self.y_true,
            'y_pred': self.y_pred
        })
        true_pred_df.to_csv(self.predictions_csv, index=False)
    
    def save_results(self):
        with open(self.results_dir, "w") as f:
            json.dump(self.results, f, indent=4)

    def save_history(self):
        history_dict = self.history.history
        history_df = pd.DataFrame(history_dict)
        history_df['training_type'] = self.test_type
        history_df['trial'] = self.trial
        history_df.to_csv(self.history_csv, index_label='epoch')

    def save_class_metrics(self):
        report = classification_report(
            self.y_true, self.y_pred, target_names=self.test_classes, output_dict=True)
        logger.info("Classification Report:", report)
        report_df = pd.DataFrame(report).transpose()
        report_df['training_type'] = self.test_type
        report_df['trial'] = self.trial
        report_df.to_csv(self.metrics_csv, sep='\t')
        
    def visualize_predictions(self, num_images=5):
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

                correct = (top3_indices[i][0] == true_labels[i])
                title_color = 'green' if correct else 'red'
                title = f"True: {true_label}\n"
                for rank in range(3):
                    label = pred_labels[rank]
                    conf = confidences[rank]
                    title += f"{rank+1}. {label} ({conf:.2%})\n"

                ax = plt.subplot(num_images, 1, i + 1)
                plt.imshow(img)
                plt.axis("off")
                ax.set_title(title, color=title_color, fontsize=10, loc='left')

            plt.tight_layout()
            plt.savefig(self.prediction_imgs_dir, bbox_inches="tight")
            break


class PlotResults():
    def __init__(self, results_dir: Path = cfg.results_dir, test_type: str = None, trial: int = None):
        self.results_dir = results_dir
        self.test_type = test_type
        self.trial = trial

        self.history_csv = cfg.get_file_name(
            cfg.training_data_dir, 'training_history', 'csv', test_type=self.test_type)
        self.history_plot = cfg.get_file_name(
            cfg.training_plots_dir, 'training_history', 'png', test_type=self.test_type)
        
        self.predictions_csv = cfg.get_file_name(
            cfg.training_data_dir, 'predictions', 'csv', test_type=self.test_type)
        self.matrix_plot = cfg.get_file_name(
            cfg.training_plots_dir, 'confusion_matrix', 'png', test_type=self.test_type)
        
        self.class_metrics_csv = cfg.get_file_name(
            cfg.class_metrics_dir, 'class_metrics', 'csv', test_type=self.test_type)
        self.class_metrics_plot = cfg.get_file_name(
            cfg.training_plots_dir, 'class_metrics', 'png', test_type=self.test_type)
        

    def plot_training_history(self):
        df = pd.read_csv(self.history_csv)

        metrics = [m.name if not isinstance( m, str) else m for m in cfg.metrics] + ['loss']
        val_metrics = [f'val_{m}' for m in metrics]

        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

        for i, (metric, val_metric) in enumerate(zip(metrics, val_metrics)):
            ax = axes[i] if num_metrics > 1 else axes
            if metric not in df.columns or val_metric not in df.columns:
                logger.warning(f"Metric '{metric}' or '{val_metric}' not found in history CSV.")
                continue
            ax.plot(df[metric], label='Train')
            ax.plot(df[val_metric], label='Val')
            ax.set_title(metric.capitalize())
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
        plt.tight_layout()
        plt.savefig(self.history_plot, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self):
        df = pd.read_csv(self.predictions_csv)
        y_true = df['y_true']
        y_pred = df['y_pred']
        cm = confusion_matrix(y_true, y_pred, labels=range(len(cfg.test_classes)))
        fig, ax = plt.subplots(figsize=(12, 10))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.test_classes)
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax, xticks_rotation=45)

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.savefig(self.matrix_plot, bbox_inches="tight")
        plt.close()

    def plot_class_metrics(self):
        df = pd.read_csv(self.class_metrics_csv, sep='\t')
        df.set_index(df.columns[0], inplace=True)
        df = df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        print(df.columns)
        df[['precision', 'recall', 'f1-score']].plot.bar(figsize=(12, 6))
        plt.title('Per-Class Metrics')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.class_metrics_plot)
        plt.close()


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
from modeling.cnn_model_func import get_optimizer


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
                       class_weights: np.ndarray = cfg.class_weights):

    for stage in range(1, cfg.num_stages + 1):
        cfg.current_stage = stage
        frozen_layers = cfg.get_stage_frozen_layers(cfg.current_stage)
        learning_rate = cfg.get_stage_lr(cfg.current_stage)

        logger.info(
            f'[Stage {stage}] \n Frozen layers: {frozen_layers}, \n Learning rate: {learning_rate:.2e}')

        base_model = model.get_layer('base_model')
        for i, layer in enumerate(base_model.layers):
            layer.trainable = i >= frozen_layers

        _run_training_stage(model,
                            train_ds,
                            val_ds,
                            test_ds,
                            class_weights,
                            stage=cfg.current_stage,
                            learning_rate=learning_rate)


def _run_training_stage(model: keras.Model,
                        train_ds: tf.data.Dataset,
                        val_ds: tf.data.Dataset,
                        test_ds: tf.data.Dataset,
                        class_weights: np.ndarray,
                        stage: int,
                        learning_rate: float):

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=cfg.patience,
                                               restore_best_weights=True),
                 keras.callbacks.ModelCheckpoint(filepath=cfg.model_checkpoint,
                                                 save_best_only=True)]

    optimizer = get_optimizer(name=cfg.optimizer,
                              lr=learning_rate,
                              use_schedule=cfg.use_schedule,
                              schedule=cfg.schedule_type,
                              first_decay_steps=cfg.first_decay_steps)

    model.compile(loss=keras.losses.CategoricalCrossentropy(
        label_smoothing=cfg.smoothing), optimizer=optimizer, metrics=cfg.metrics)

    # FIX for fine tuning base model
    visualize_model(model)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    results = model.evaluate(test_ds, return_dict=True)
    logger.info(f'[Stage {stage}] Test results: {results}')

    evaluator = ModelEvaluator(
        model, test_ds, history=history, results=results)

    evaluator.save_true_pred_vals()
    evaluator.save_results()
    evaluator.save_history()
    evaluator.save_class_metrics()
    evaluator.visualize_predictions(num_images=5)


class ModelEvaluator:
    def __init__(self,
                 model: keras.Model,
                 test_ds: tf.data.Dataset,
                 history: dict = None,
                 results: dict = None):
        self.model = model
        self.test_ds = test_ds
        self.history = history
        self.results = results
        self.y_true = None
        self.y_pred = None

    def save_true_pred_vals(self):
        if self.y_true is None or self.y_pred is None:
            predictions = self.model.predict(self.test_ds)
            self.y_pred = np.argmax(predictions, axis=1)
            self.y_true = np.concatenate([np.argmax(labels.numpy(), axis=1)
                                          for _, labels in self.test_ds])
        df = pd.DataFrame({'y_true': self.y_true,
                           'y_pred': self.y_pred})
        df.to_csv(cfg.predictions_csv, index=False)

    def save_results(self):
        with open(cfg.json_results, 'w') as f:
            json.dump(self.results, f, indent=4)

    def save_history(self):
        history_dict = self.history.history
        df = pd.DataFrame(history_dict)
        df['training_stage'] = cfg.current_stage
        df['trial'] = cfg.trial_num
        df.to_csv(cfg.history_csv, index_label='epoch')

    def save_class_metrics(self):
        report = classification_report(self.y_true,
                                       self.y_pred,
                                       target_names=cfg.test_classes,
                                       output_dict=True)
        logger.info('Classification Report:\n%s', json.dumps(report, indent=2))
        df = pd.DataFrame(report).transpose()
        df['training_stage'] = cfg.current_stage
        df['trial'] = cfg.trial_num
        df.to_csv(cfg.class_metrics_csv)

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
                title = f'True: {true_label}\n'
                for rank in range(3):
                    label = pred_labels[rank]
                    conf = confidences[rank]
                    title += f'{rank+1}. {label} ({conf:.2%})\n'

                ax = plt.subplot(num_images, 1, i + 1)
                plt.imshow(img)
                plt.axis('off')
                ax.set_title(title, color=title_color, fontsize=10, loc='left')

            plt.tight_layout()
            plt.savefig(cfg.prediction_visualization_plot, bbox_inches='tight')
            break


class PlotResults():
    def __init__(self, test_type: str = None, trial: int = None):
        self.test_type = test_type
        self.trial = trial

    def plot_training_history(self):
        df = pd.read_csv(cfg.history_csv)

        metrics = [m.name if not isinstance(
            m, str) else m for m in cfg.metrics] + ['loss']
        val_metrics = [f'val_{m}' for m in metrics]

        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

        for i, (metric, val_metric) in enumerate(zip(metrics, val_metrics)):
            ax = axes[i] if num_metrics > 1 else axes
            if metric not in df.columns or val_metric not in df.columns:
                logger.warning(
                    f"Metric '{metric}' or '{val_metric}' not found in history CSV.")
                continue
            ax.plot(df[metric], label='Train')
            ax.plot(df[val_metric], label='Val')
            ax.set_title(metric.capitalize())
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
        plt.tight_layout()
        plt.savefig(cfg.history_plot, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self):
        df = pd.read_csv(cfg.predictions_csv)
        y_true = df['y_true']
        y_pred = df['y_pred']
        cm = confusion_matrix(
            y_true, y_pred, labels=range(len(cfg.test_classes)))
        fig, ax = plt.subplots(figsize=(12, 10))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=cfg.test_classes)
        disp.plot(cmap=plt.cm.Blues, values_format='d',
                  ax=ax, xticks_rotation=45)

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.savefig(cfg.confusion_matrix_plot, bbox_inches='tight')
        plt.close()

    def plot_class_metrics(self):
        df = pd.read_csv(cfg.class_metrics_csv)
        df.set_index(df.columns[0], inplace=True)
        df = df.drop(index=['accuracy', 'macro avg',
                     'weighted avg'], errors='ignore')
        print(df.columns)
        df[['precision', 'recall', 'f1-score']].plot.bar(figsize=(12, 6))
        plt.title('Per-Class Metrics')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(cfg.class_metrics_plot)
        plt.close()


def visualize_model(model):
    keras.utils.plot_model(model,
                           to_file=f'model_stage{cfg.current_stage}.png',
                           show_shapes=False,
                           show_dtype=False,
                           show_layer_names=True,
                           rankdir='LR',
                           expand_nested=False,
                           dpi=200,
                           show_layer_activations=True,
                           show_trainable=True,
                           )

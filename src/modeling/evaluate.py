from config import Config
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import top_k_categorical_accuracy
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
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=cfg.topk)]
    )

    # Print model summary
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    results = model.evaluate(test_ds, return_dict=True)

    with open(cfg.training_history_dir / f"test_metrics_trial_{trial}_{test_type}.json", "w") as f:
        json.dump(results, f, indent=4)

    evaluator = ModelEvaluator(
        model, test_ds, trial=trial, test_type=test_type)

    evaluator.plot_history(history)
    evaluator.plot_confusion_matrix()
    evaluator.plot_class_metrics()
    evaluator.save_history(history, cfg.training_history_dir /
                           f"history_trial_{trial}_{test_type}.json")
    evaluator.visualize_predictions()

    return history


class ModelEvaluator:
    def __init__(self, model, test_ds, trial: int = None, results_dir: Path = cfg.results_dir, test_type: str = 'test'):
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

    def plot_history(self, history):
        file_name = f'training_history_trial_{cfg.trial_num}_{self.test_type}.png'

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        top_k_acc = history.history.get('top_k_categorical_accuracy', None)
        val_top_k_acc = history.history.get(
            'val_top_k_categorical_accuracy', None)
        epochs = range(1, len(acc) + 1)

        n_subplots = 3 if top_k_acc and val_top_k_acc else 2
        fig, axs = plt.subplots(1, n_subplots, figsize=(6*n_subplots, 5))

        axs[0].plot(epochs, acc, 'r', label='Training')
        axs[0].plot(epochs, val_acc, 'b', label='Validation')
        axs[0].set_title('Model Accuracy')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(epochs, loss, 'r', label='Training')
        axs[1].plot(epochs, val_loss, 'b', label='Validation')
        axs[1].set_title('Model Loss')
        axs[1].grid(True)
        axs[1].legend()

        if n_subplots == 3:
            axs[2].plot(epochs, top_k_acc, 'r', label='Training')
            axs[2].plot(epochs, val_top_k_acc, 'b',
                        label='Validation')
            axs[2].set_title(f'Model top-k Accuracy (k={cfg.topk})')
            axs[2].grid(True)
            axs[2].legend()

        plt.tight_layout()
        plt.savefig(cfg.training_history_dir / file_name, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self):
        file_name = f'confusion_matrix_trial_{cfg.trial_num}_{self.test_type}.png'

        y_true, y_pred = self.get_true_pred_vals()

        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.test_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.test_classes)
        disp.plot(xticks_rotation=45)

        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(cfg.confusion_matrix_dir / file_name, bbox_inches="tight")
        plt.close()

    def plot_class_metrics(self):
        file_name = f'class_metrics_trial_{cfg.trial_num}_{self.test_type}.csv'
        y_true, y_pred = self.get_true_pred_vals()
        report = classification_report(
            y_true, y_pred, target_names=self.test_classes, output_dict=True)
        print("Classification Report:", report)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(cfg.class_metrics_dir / file_name, sep='\t')

    def save_history(self, history, history_file: Path):
        history_dict = history.history
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=4)

    def visualize_predictions(self, num_images=5):
        # Take a batch from the dataset
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
            plt.savefig(cfg.confusion_matrix_dir /
                        f'predictions_trial_{cfg.trial_num}_{self.test_type}.png', bbox_inches="tight")
            break  # Only one batch

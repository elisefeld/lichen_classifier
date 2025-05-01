import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from config import Config
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import top_k_categorical_accuracy

cfg = Config()


def train_and_evaluate(model,
                       train_ds,
                       val_ds,
                       test_ds,
                       optimizer,
                       callbacks,
                       class_names,
                       class_weights,
                       trial: int = None):
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.smoothing),
        optimizer=optimizer,
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=cfg.topk)]
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    model.evaluate(test_ds)
    evaluator = ModelEvaluator(model, test_ds, class_names, trial=trial)
    evaluator.plot_history(history)
    evaluator.plot_learning_rate(optimizer.learning_rate, epochs=cfg.epochs)
    evaluator.plot_confusion_matrix()
    evaluator.plot_class_metrics()
    return history


class ModelEvaluator:
    def __init__(self, model, test_ds, test_classes, trial: int = None, results_dir: Path = cfg.results_dir):
        self.model = model
        self.test_ds = test_ds
        self.test_classes = test_classes
        self.results_dir = results_dir
        self.y_true = None
        self.y_pred = None
        self.trial = trial

    def get_true_pred_vals(self):
        predictions = self.model.predict(self.test_ds)
        self.y_pred = np.argmax(predictions, axis=1)
        self.y_true = np.concatenate([np.argmax(labels.numpy(), axis=1)
                                      for _, labels in self.test_ds])
        return self.y_true, self.y_pred

    def plot_history(self, history, save: bool = True, file_name: str = None):
        if file_name is None:
            file_name = f'history_{self.trial}.png'
        else:
            file_name = f'{file_name}_{self.trial}.png'

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

        axs[0].plot(epochs, acc, 'r', label='Training acc')
        axs[0].plot(epochs, val_acc, 'b', label='Validation acc')
        axs[0].set_title('Accuracy')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(epochs, loss, 'r', label='Training loss')
        axs[1].plot(epochs, val_loss, 'b', label='Validation loss')
        axs[1].set_title('Loss')
        axs[1].grid(True)
        axs[1].legend()

        if n_subplots == 3:
            axs[2].plot(epochs, top_k_acc, 'r', label='Training top-k acc')
            axs[2].plot(epochs, val_top_k_acc, 'b',
                        label='Validation top-k acc')
            axs[2].set_title(f'Top-k accuracy (k={cfg.topk})')
            axs[2].grid(True)
            axs[2].legend()

        plt.tight_layout()
        if save:
            plt.savefig(self.results_dir / file_name)
        plt.show()

    def plot_learning_rate(self, lr_schedule, epochs, save: bool = True, file_name: str = None):
        if file_name is None:
            file_name = f'learning_rate_{self.trial}.png'
        else:
            file_name = f'{file_name}_{self.trial}.png'

        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), lr_schedule)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        if save:
            plt.savefig(self.results_dir / file_name)
        plt.show()

    def plot_confusion_matrix(self, save: bool = True, file_name: str = None):
        if file_name is None:
            file_name = f'confusion_matrix_{self.trial}.png'
        else:
            file_name = f'{file_name}_{self.trial}.png'
        y_true, y_pred = self.get_true_pred_vals()
        cm = confusion_matrix(
            y_true, y_pred, labels=range(len(self.test_classes)))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.test_classes)
        disp.plot(xticks_rotation=90)
        if save:
            plt.savefig(self.results_dir / file_name)
        plt.show()
        return cm

    def plot_class_metrics(self):
        '''Prints per-class precision, recall, F1'''
        y_true, y_pred = self.get_true_pred_vals()
        return classification_report(y_true, y_pred, target_names=self.test_classes, output_dict=True)

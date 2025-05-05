from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import top_k_categorical_accuracy

from config import Config

cfg = Config()


def train_and_evaluate(model: keras.Model,
                       train_ds: tf.data.Dataset,
                       val_ds: tf.data.Dataset,
                       test_ds: tf.data.Dataset,
                       optimizer: keras.optimizers.Optimizer,
                       callbacks: list,
                       class_names: list,
                       class_weights: np.ndarray = None,
                       trial: int = None):
    '''
    Trains and evaluates a given model using the provided datasets, optimizer, 
    and callbacks. Additionally, it generates evaluation plots such as training 
    history, confusion matrix, and class-specific metrics.

    Args:
        model (keras.Model): The model to be trained and evaluated.
        train_ds (tf.data.Dataset): The dataset used for training.
        val_ds (tf.data.Dataset): The dataset used for validation during training.
        test_ds (tf.data.Dataset): The dataset used for final evaluation.
        optimizer (keras.optimizers.Optimizer): The optimizer to use for training.
        callbacks (list): A list of Keras callbacks to use during training.
        class_names (list): A list of class names corresponding to the dataset labels.
        class_weights (dict): A dictionary mapping class indices to weights for 
                                handling class imbalance.
        trial (int, optional): An optional trial identifier for tracking experiments. 
                                Defaults to None.

    Returns:
        keras.callbacks.History: The training history object containing details 
                                    about the training process.
    '''
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
        '''
        Computes and returns a classification report with per-class metrics.

        This method calculates precision, recall, and F1-score for each class
        based on the true and predicted values obtained from the model. The
        results are returned as a dictionary.

        Returns:
            dict: A dictionary containing precision, recall, F1-score, and 
                  support for each class, as well as overall metrics.
        '''
        y_true, y_pred = self.get_true_pred_vals()
        report = classification_report(
            y_true, y_pred, target_names=self.test_classes, output_dict=True)
        print("Classification Report:", report)
        return report

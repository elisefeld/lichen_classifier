from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import utils
import classes
from config import Config

cfg = Config()
keras.backend.clear_session()
mixed_precision.set_global_policy('mixed_float16')

##############################
######### SPLIT DATA #########
##############################
train_ds = utils.load_dataset(cfg.train_dir, cfg.batch_size, cfg.crop_dim)
val_ds = utils.load_dataset(cfg.val_dir, cfg.batch_size, cfg.crop_dim)
test_ds = utils.load_dataset(cfg.test_dir, cfg.batch_size, cfg.crop_dim)

test_classes = test_ds.class_names
y_train = np.concatenate([np.argmax(y.numpy(), axis=1)
                          for _, y in train_ds])
num_classes = len(np.unique(y_train))
class_weights = dict(enumerate(
    class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train)))

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

#############################
########## TUNING  ##########
#############################
optimizer = utils.get_optimizer(optimizer=cfg.optimizer,
                                initial_learning_rate=cfg.learning_rate,
                                use_schedule=cfg.use_schedule,
                                schedule_type=cfg.schedule_type,
                                decay_steps=cfg.decay_steps,
                                decay_rate=cfg.decay_rate,
                                staircase=True)

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=cfg.patience,
                                           restore_best_weights=True),
             keras.callbacks.ModelCheckpoint(f'best_model_{cfg.base_model.lower()}_{cfg.optimizer.lower()}.keras',
                                             save_best_only=True)]

##############################
########### MODEL  ###########
##############################
model = classes.LichenClassifier(seed=cfg.seed,
                                 factor=cfg.transform_factor,
                                 dim=cfg.dim,
                                 crop_dim=cfg.crop_dim,
                                 base_model=cfg.base_model,
                                 frozen_layers=cfg.frozen_layers,
                                 num_classes=num_classes)

##############################
########## TRAINING ##########
##############################
model.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              optimizer=optimizer,
              metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2)])

model.summary()

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=cfg.epochs,
                    callbacks=callbacks,
                    class_weight=class_weights)

##############################
######### EVALUATION #########
##############################
model.evaluate(test_ds)

# Plot training history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# Confusion matrix
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate([np.argmax(labels.numpy(), axis=1)
                        for _, labels in test_ds])

cm = confusion_matrix(y_true, y_pred, labels=range(len(test_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_classes)
disp.plot(xticks_rotation=90)
plt.show()

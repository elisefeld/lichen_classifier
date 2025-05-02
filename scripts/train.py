
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

import utils
import modeling.cnn_model as cnn_model
from modeling.cnn_model import get_optimizer
from modeling.evaluate import train_and_evaluate
import visualization.img_plots
from config import Config

cfg = Config()

POLICY = 'mixed_float16' if cfg.mixed_precision else 'float32'
mixed_precision.set_global_policy(POLICY)
keras.backend.clear_session()

##############################
######### SPLIT DATA #########
##############################
train_ds = utils.img_data.load_img_dataset(cfg.train_dir, cfg.batch_size, cfg.crop_dim)
val_ds = utils.img_data.load_img_dataset(cfg.val_dir, cfg.batch_size, cfg.crop_dim)
test_ds = utils.img_data.load_img_dataset(cfg.test_dir, cfg.batch_size, cfg.crop_dim)

class_names, num_classes, class_weights = utils.img_data.get_class_info(train_ds)

visualization.img_plots.rgb_histograms_grid(train_ds, class_names)

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

#############################
########## TUNING  ##########
#############################
optimizer = get_optimizer(optimizer=cfg.optimizer,
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
model = cnn_model.LichenClassifier(seed=cfg.seed,
                                 factor=cfg.transform_factor,
                                 dim=cfg.dim,
                                 crop_dim=cfg.crop_dim,
                                 base_model=cfg.base_model,
                                 frozen_layers=cfg.frozen_layers,
                                 num_classes=num_classes)

##############################
########## TRAINING ##########
##############################
model.freeze_base_model()

coarse_history = train_and_evaluate(model=model,
                                    train_ds=train_ds,
                                    val_ds=val_ds,
                                    test_ds=test_ds,
                                    optimizer=optimizer,
                                    callbacks=callbacks,
                                    class_names=class_names,
                                    class_weights=class_weights,
                                    trial=1)

model.unfreeze_base_model(None)

learning_rate = cfg.learning_rate * 0.1  # e.g. 5e-5
optimizer = get_optimizer(optimizer=cfg.optimizer,
                            initial_learning_rate=learning_rate,
                            use_schedule=cfg.use_schedule,
                            schedule_type=cfg.schedule_type,
                            decay_steps=cfg.decay_steps,
                            decay_rate=cfg.decay_rate,
                            staircase=True)


fine_history = train_and_evaluate(model=model,
                                  train_ds=train_ds,
                                  val_ds=val_ds,
                                  test_ds=test_ds,
                                  optimizer=optimizer, 
                                  callbacks=callbacks,
                                  class_names=class_names,
                                  class_weights=class_weights,
                                  trial=2)

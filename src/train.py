import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from utils.data import load_img_dataset, get_class_info
from utils.plotting import rgb_histograms_grid
from modeling.cnn_model import LichenClassifier, get_optimizer
from modeling.evaluate import train_and_evaluate

from config import Config
cfg = Config()

# Set random seeds
tf.random.set_seed(cfg.seed)

POLICY = 'mixed_float16' if cfg.mixed_precision else 'float32'
mixed_precision.set_global_policy(POLICY)
tf.keras.backend.clear_session()

# Load and preprocess image data
train_ds = load_img_dataset(cfg.train_dir, cfg.batch_size, cfg.crop_dim)
val_ds = load_img_dataset(cfg.val_dir, cfg.batch_size, cfg.crop_dim)
test_ds = load_img_dataset(cfg.test_dir, cfg.batch_size, cfg.crop_dim)

# Get class information
class_names, num_classes, class_weights = get_class_info(train_ds)

if cfg.plot_imgs:
    rgb_histograms_grid(train_ds, class_names)

# Prefetching and caching
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# Optimizer and callbacks
optimizer = get_optimizer(name=cfg.optimizer,
                          lr=cfg.learning_rate,
                          use_schedule=cfg.use_schedule,
                          schedule=cfg.schedule_type,
                          decay_steps=cfg.decay_steps,
                          decay_rate=cfg.decay_rate,
                          staircase=True)

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=cfg.patience,
                                           restore_best_weights=True),
             keras.callbacks.ModelCheckpoint(f'best_model_{cfg.base_model.lower()}_{cfg.optimizer.lower()}.keras',  # improve the file name
                                             save_best_only=True)]  # create a function for this

# Initialize Model
model = LichenClassifier(rotation=cfg.rotation_factor,
                         contrast=cfg.contrast_factor,
                         translation=cfg.translation_factor,
                         dim=cfg.dim,
                         crop_dim=cfg.crop_dim,
                         base_model=cfg.base_model,
                         num_classes=num_classes)

# Stage 1: Train the model with frozen base model (coarse training)
model.freeze_base_model()
coarse_history = train_and_evaluate(model=model,
                                    train_ds=train_ds,
                                    val_ds=val_ds,
                                    test_ds=test_ds,
                                    optimizer=optimizer,
                                    callbacks=callbacks,
                                    class_names=class_names,
                                    class_weights=class_weights,
                                    trial=1)  # edit this so it saves the graphs to results dir

# Stage 2: Train the model with unfrozen base model (fine-tuning)
model.unfreeze_base_model(None)
learning_rate = cfg.learning_rate * 0.1
optimizer = get_optimizer(name=cfg.optimizer,
                          lr=learning_rate,
                          use_schedule=cfg.use_schedule,
                          schedule=cfg.schedule_type,
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

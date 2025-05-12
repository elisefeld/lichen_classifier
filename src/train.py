import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from utils.data import load_img_dataset, get_class_info
from utils.plotting import plot_rgb_histograms
from modeling.cnn_model_func import build_lichen_classifier, get_optimizer
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
    plot_rgb_histograms(train_ds, class_names)

# Prefetching and caching
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)


# Initialize Model
model = build_lichen_classifier(input_shape=(224, 224, 3),
                                dim=cfg.dim,
                                crop_dim=cfg.crop_dim,
                                rotation=cfg.rotation_factor,
                                contrast=cfg.contrast_factor,
                                translation=cfg.translation_factor,
                                base_model_name=cfg.base_model,
                                num_classes=num_classes)

# Stage 1: Train the model with frozen base model (coarse training)
optimizer = get_optimizer(name=cfg.optimizer,
                          lr=cfg.coarse_learning_rate,
                          use_schedule=cfg.use_schedule,
                          schedule=cfg.schedule_type,
                          first_decay_steps=cfg.first_decay_steps)
coarse_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=cfg.patience,
                                                  restore_best_weights=True),
                    keras.callbacks.ModelCheckpoint(cfg.model_dir / f'coarse_{cfg.base_model.lower()}_{cfg.optimizer.lower()}_trial_{cfg.trial_num}.keras',
                                                    save_best_only=True)] # FIX add set path here

for layer in model.get_layer("base_model").layers:
    layer.trainable = False

coarse_history = train_and_evaluate(model=model,
                                    train_ds=train_ds,
                                    val_ds=val_ds,
                                    test_ds=test_ds,
                                    optimizer=optimizer,
                                    callbacks=coarse_callbacks,
                                    class_weights=class_weights,
                                    test_type='coarse',
                                    trial=cfg.trial_num)

# Stage 2: Train the model with unfrozen base model (fine-tuning)
if cfg.fine_tune:
    fine_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=cfg.patience,
                                                    restore_best_weights=True),
                    keras.callbacks.ModelCheckpoint(cfg.model_dir/f'fine_{cfg.base_model.lower()}_{cfg.optimizer.lower()}_trial_{cfg.trial_num}.keras',
                                                    save_best_only=True)] #FIX add set path here

    optimizer = get_optimizer(name=cfg.optimizer,
                             lr=cfg.fine_learning_rate,
                             use_schedule=cfg.use_schedule,
                             schedule=cfg.schedule_type,
                             first_decay_steps=cfg.first_decay_steps)

    unfreeze_layers = cfg.frozen_layers
    for layer in model.get_layer("base_model").layers[-unfreeze_layers:]:
        layer.trainable = True

    fine_history = train_and_evaluate(model=model,
                                    train_ds=train_ds,
                                    val_ds=val_ds,
                                    test_ds=test_ds,
                                    optimizer=optimizer,
                                    callbacks=fine_callbacks,
                                    class_weights=class_weights,
                                    test_type='fine_tune',
                                    trial=cfg.trial_num)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from utils.data import load_img_dataset
from utils.plotting import plot_rgb_histograms
from modeling.cnn_model_func import build_lichen_classifier
from modeling.evaluate import train_and_evaluate

from config import Config
cfg = Config()

# Set random seeds
tf.random.set_seed(cfg.seed)

POLICY = 'mixed_float16' if cfg.mixed_precision else 'float32'
mixed_precision.set_global_policy(POLICY)
tf.keras.backend.clear_session()

# Load and preprocess image data
train_ds = load_img_dataset(cfg.train_dir)
val_ds = load_img_dataset(cfg.val_dir)
test_ds = load_img_dataset(cfg.test_dir)

if cfg.plot_imgs:
    plot_rgb_histograms(train_ds, cfg.test_classes)

# Prefetching and caching
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# Initialize Model
model = build_lichen_classifier()

model.summary()

train_and_evaluate(model=model,
                   train_ds=train_ds,
                   val_ds=val_ds,
                   test_ds=test_ds)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet152, EfficientNetB0, EfficientNetV2B0
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras import optimizers

from config import Config
cfg = Config()

# Set random seeds
tf.random.set_seed(cfg.seed)

# Dictionaries
PREPROCESS_MAP = {
    'ResNet50': resnet_preprocess,
    'ResNet50V2': resnet_preprocess,
    'ResNet101': resnet_preprocess,
    'ResNet152': resnet_preprocess,
    'EfficientNetB0': efficientnet_preprocess,
    'EfficientNetV2B0': efficientnet_preprocess
}

MODEL_DICT = {
    'ResNet50': ResNet50,
    'ResNet50V2': ResNet50V2,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetV2B0': EfficientNetV2B0
}

OPT_DICT = {
    'adam': optimizers.Adam,
    'sgd': optimizers.SGD,
    'rmsprop': optimizers.RMSprop,
    'adagrad': optimizers.Adagrad,
    'adamax': optimizers.Adamax,
    'nadam': optimizers.Nadam
}

### Functions ###


def get_base_model(model_name: str) -> keras.Model:
    if model_name not in MODEL_DICT:
        raise ValueError(f'Invalid model name: {model_name}. '
                         f'Choose from {list(MODEL_DICT.keys())}.')
    return MODEL_DICT[model_name](include_top=False, weights='imagenet', input_shape=cfg.input_shape)


class CommonPreprocessing(keras.layers.Layer):
    def __init__(self, dim: int, crop_dim: int):
        super().__init__()
        self.dim = dim
        self.crop_dim = crop_dim
        self.common_preprocess = tf.keras.Sequential([
            tf.keras.layers.Resizing(
                cfg.dim, cfg.dim, crop_to_aspect_ratio=True),
            tf.keras.layers.CenterCrop(self.crop_dim, self.crop_dim)
        ])

    def call(self, inputs):
        return self.common_preprocess(inputs)


class AugmentLayer(keras.layers.Layer):
    def __init__(self,
                 rotation: float,
                 contrast: float,
                 translation: float):
        '''A custom Keras layer for applying data augmentation to input images.'''
        super().__init__()
        self.augment = keras.Sequential([
            keras.layers.RandomFlip('horizontal'),
            keras.layers.RandomRotation(rotation),
            keras.layers.RandomContrast(contrast),
            keras.layers.RandomTranslation(translation, translation)
        ])

    def call(self, inputs, training: bool):
        return self.augment(inputs, training=training) if training else inputs


class LichenClassifier(keras.Model):
    def __init__(self,
                 dim: int,
                 crop_dim: int,
                 rotation: float,
                 contrast: float,
                 translation: float,
                 base_model: str,
                 num_classes: int):
        super().__init__()
        self.preprocessing_layer = keras.layers.Lambda(
            PREPROCESS_MAP[base_model])
        self.common_preprocessing = CommonPreprocessing(
            dim=dim, crop_dim=crop_dim)
        self.augmentation = AugmentLayer(rotation=rotation, contrast=contrast,
                                         translation=translation)
        self.base_model = get_base_model(base_model)
        self.pooling = keras.layers.GlobalMaxPooling2D()
        self.custom_layers = keras.Sequential([
            keras.layers.Dense(1024, activation='selu',
                               kernel_initializer='lecun_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation='selu',
                               kernel_initializer='lecun_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='selu',
                               kernel_initializer='lecun_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1)
        ])
        self.output_layer = keras.layers.Dense(
            num_classes, activation='softmax', dtype='float32')

    def call(self, inputs, training: bool):
        x = self.preprocessing_layer(inputs)
        x = self.common_preprocessing(x)
        x = self.augmentation(x, training=training)
        x = self.base_model(x, training=training)
        x = self.pooling(x)
        x = self.custom_layers(x, training=training)
        x = self.output_layer(x, training=training)
        return x

    def freeze_base_model(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def unfreeze_base_model(self, unfreeze_layers=None):
        self.base_model.trainable = True
        if unfreeze_layers is not None:
            for layer in self.base_model.layers[:-unfreeze_layers]:
                layer.trainable = False


def get_optimizer(name: str = 'adam',
                  lr: float = 1e-3,
                  schedule: str = 'exponential',
                  decay_steps: int = 1000,
                  first_decay_steps: int = 10000,
                  decay_rate: float = 0.9,
                  use_schedule: bool = True,
                  staircase: bool = True,
                  t_mul: float = 1.0,
                  m_mul: float = 1.0,
                  alpha: float = 0.0) -> optimizers.Optimizer:
    if use_schedule:
        if schedule == 'exponential':
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                decay_steps=decay_steps,
                                                                decay_rate=decay_rate,
                                                                staircase=staircase)
        elif schedule == 'cosine':
            lr_schedule = optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr,
                                                                   first_decay_steps=first_decay_steps,
                                                                   t_mul=t_mul,
                                                                   m_mul=m_mul,
                                                                   alpha=alpha)
        else:
            raise ValueError(
                "Invalid schedule type. Choose from 'exponential' or 'cosine'.")
    else:
        lr_schedule = lr

    if name.lower() not in OPT_DICT:
        raise ValueError(
            f"Invalid optimizer name: {name}. Choose from {list(OPT_DICT.keys())}.")
    return OPT_DICT[name](learning_rate=lr_schedule)

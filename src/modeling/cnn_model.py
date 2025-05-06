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
def get_base_model(model_name: str):
    if model_name not in MODEL_DICT:
        raise ValueError(f'Invalid model name: {model_name}. '
                         f'Choose from {list(MODEL_DICT.keys())}.')
    return MODEL_DICT[model_name](include_top=False, weights='imagenet')


class AugmentLayer(keras.layers.Layer):
    def __init__(self,
                 rotation: float = 0.2,
                 contrast: float = 0.2,
                 translation: float = 0.2,
                 dim: int = 256,
                 crop_dim: int = 224):
        '''A custom Keras layer for applying data augmentation to input images.'''
        super().__init__()
        self.augment = keras.Sequential([
            keras.layers.Resizing(dim, dim, crop_to_aspect_ratio=True),
            keras.layers.RandomCrop(crop_dim, crop_dim),
            keras.layers.RandomFlip('horizontal'),
            keras.layers.RandomRotation(rotation),
            keras.layers.RandomContrast(contrast),
            keras.layers.RandomTranslation(translation, translation)
        ])

    def call(self, inputs, training=False):
        return self.augment(inputs, training=training) if training else inputs


class LichenClassifier(keras.Model):
    def __init__(self,
                 num_classes: int,
                 base_model: str = 'ResNet50',
                 dim: int = 256,
                 crop_dim: int = 224,
                 rotation: float = 0.2,
                 contrast: float = 0.2,
                 translation: float = 0.2

                 ):
        super().__init__()
        self.preprocessing_layer = keras.layers.Lambda(
            PREPROCESS_MAP[base_model])
        self.augmentation = AugmentLayer(rotation=rotation, contrast=contrast,
                                         translation=translation, dim=dim, crop_dim=crop_dim)
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

    def call(self, inputs, training=False):
        x = self.preprocessing_layer(inputs)
        x = self.augmentation(x, training=training)
        x = self.base_model(x, training=training)
        x = self.pooling(x)
        x = self.custom_layers(x, training=training)
        x = self.output_layer(x, training=training)
        return x

    def freeze_base_model(self):
        '''
        Freezes the layers of the base model to make them non-trainable.

        This method iterates through all the layers of the base model and sets 
        their `trainable` attribute to `False`. This used to prevent 
        the base model's weights from being updated during training, allowing 
        only the newly added layers to be trained.

        Returns:
            None
        '''
        for layer in self.base_model.layers:
            layer.trainable = False

    def unfreeze_base_model(self, frozen_layers: int = None):
        '''
        Unfreezes layers of the base model for training.

        This method sets the `trainable` attribute of the layers in the base model.
        Layers with indices less than `frozen_layers` will remain frozen (non-trainable),
        while the rest will be unfrozen (trainable).

        Args:
            frozen_layers (int, optional): The number of layers to keep frozen. 
                If None, all layers will be unfrozen. Defaults to None.
        '''
        for i, layer in enumerate(self.base_model.layers):
            if frozen_layers is not None and i < frozen_layers:
                layer.trainable = False
            else:
                layer.trainable = True


def get_optimizer(name: str = 'adam',
                  lr: float = 1e-3,
                  schedule: str = 'exponential',
                  decay_steps: int = 1000,
                  decay_rate: float = 0.9,
                  use_schedule: bool = True,
                  staircase: bool = True,
                  **kwargs):
    if use_schedule:
        if schedule == 'exponential':
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                decay_steps=decay_steps,
                                                                decay_rate=decay_rate,
                                                                staircase=staircase)
        elif schedule == 'cosine':
            lr_schedule = optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr,
                                                                   first_decay_steps=decay_steps)
        else:
            raise ValueError(
                "Invalid schedule type. Choose from 'exponential' or 'cosine'.")
    else:
        lr_schedule = lr

    if name is not None and isinstance(name, str):
        name = name.lower()

    if name not in OPT_DICT:
        raise ValueError(
            f"Invalid optimizer name: {name}. Choose from {list(OPT_DICT.keys())}.")
    return OPT_DICT[name](learning_rate=lr_schedule, **kwargs)

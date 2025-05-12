import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet152, EfficientNetB0, EfficientNetV2B0
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras import optimizers

from config import Config
cfg = Config()

tf.random.set_seed(cfg.seed)

# Dictionaries
MODEL_MAP = {
    'ResNet50': {'model': ResNet50, 'preprocess': resnet_preprocess},
    'ResNet50V2': {'model': ResNet50V2, 'preprocess': resnet_preprocess},
    'ResNet101': {'model': ResNet101, 'preprocess': resnet_preprocess},
    'ResNet152': {'model': ResNet152, 'preprocess': resnet_preprocess},
    'EfficientNetB0': {'model': EfficientNetB0, 'preprocess': efficientnet_preprocess},
    'EfficientNetV2B0': {'model': EfficientNetV2B0, 'preprocess': efficientnet_preprocess}
}

OPT_DICT = {
    'adam': optimizers.Adam,
    'sgd': optimizers.SGD,
    'rmsprop': optimizers.RMSprop,
    'adagrad': optimizers.Adagrad,
    'adamax': optimizers.Adamax,
    'nadam': optimizers.Nadam
}


def build_lichen_classifier(input_shape: tuple,
                            dim: int,
                            crop_dim: int,
                            rotation: float,
                            contrast: float,
                            translation: float,
                            base_model_name: str,
                            num_classes: int) -> keras.Model:

    inputs = keras.Input(shape=input_shape, name='input_layer')

    x = keras.layers.Lambda(
        MODEL_MAP[base_model_name]['preprocess'], name='preprocess')(inputs)
    x = keras.layers.Resizing(
        dim, dim, crop_to_aspect_ratio=True, name='resize')(x)
    x = keras.layers.CenterCrop(crop_dim, crop_dim, name='center_crop')(x)

    x = keras.layers.RandomFlip(
        'horizontal', name='random_flip')(x, training=True)
    x = keras.layers.RandomRotation(
        rotation, name='random_rotation')(x, training=True)
    x = keras.layers.RandomContrast(
        contrast, name='random_contrast')(x, training=True)
    x = keras.layers.RandomTranslation(
        translation, translation, name='random_translation')(x, training=True)

    base_model = MODEL_MAP[base_model_name]['model'](
        include_top=False,
        weights='imagenet',
        input_shape=(crop_dim, crop_dim, 3),
        name='base_model'
    )
    x = base_model(x)
    x = keras.layers.GlobalMaxPooling2D(name='global_max_pool')(x)

    x = keras.layers.Dense(1024, activation='selu',
                           kernel_initializer='lecun_normal', name='dense_1024')(x)
    x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
    x = keras.layers.Dropout(0.3, name='dropout_1')(x)

    x = keras.layers.Dense(
        512, activation='selu', kernel_initializer='lecun_normal', name='dense_512')(x)
    x = keras.layers.BatchNormalization(name='batch_norm_2')(x)
    x = keras.layers.Dropout(0.2, name='dropout_2')(x)

    x = keras.layers.Dense(
        128, activation='selu', kernel_initializer='lecun_normal', name='dense_128')(x)
    x = keras.layers.BatchNormalization(name='batch_norm_3')(x)
    x = keras.layers.Dropout(0.1, name='dropout_3')(x)

    outputs = keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32', name='output_layer')(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name='lichen_classifier')
    return model


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

    if name.lower() not in OPT_DICT:
        raise ValueError(
            f"Invalid optimizer name: {name}. Choose from {list(OPT_DICT.keys())}.")

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
                "Invalid schedule type. Choose from 'exponential' or 'cosine'")
    else:
        lr_schedule = lr

    return OPT_DICT[name](learning_rate=lr_schedule)

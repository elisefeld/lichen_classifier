import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input
import utils


def get_base_model(model_name):
    model_dict = {
        'ResNet50': ResNet50,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152,
        'EfficientNetB0': EfficientNetB0}
    if model_name in model_dict:
        return model_dict[model_name](include_top=False, weights='imagenet')
    raise ValueError("Invalid model name: {model_name}. Choose from {model_dict.keys()}.")


class AugmentLayer(keras.layers.Layer):
    def __init__(self, seed, factor=0.2, dim=256, crop_dim=224):
        super().__init__()
        self.augment = keras.Sequential([
            keras.layers.Resizing(dim, dim, crop_to_aspect_ratio=True),
            keras.layers.RandomCrop(crop_dim, crop_dim, seed=seed),
            keras.layers.RandomFlip('horizontal', seed=seed),
            keras.layers.RandomRotation(factor, seed=seed),
            keras.layers.RandomContrast(factor, seed=seed),
            keras.layers.RandomTranslation(factor, factor, seed=seed)
        ])

    def call(self, inputs, training=False):
        return self.augment(inputs, training=training)


class LichenClassifier(keras.Model):
    def __init__(self, seed, factor, dim, crop_dim, base_model, frozen_layers, num_classes):
        super().__init__()
        self.inputs = keras.Input(
            shape=(crop_dim, crop_dim, 3), dtype='float32')
        self.preprocess = preprocess_input
        self.augmentation = AugmentLayer(
            seed=seed, factor=factor, dim=dim, crop_dim=crop_dim)
        self.base_model = get_base_model(base_model)
        for layer in self.base_model.layers[:frozen_layers]:
            layer.trainable = False
        for layer in self.base_model.layers[frozen_layers:]:
            layer.trainable = True
        self.pooling = keras.layers.GlobalMaxPooling2D()
        self.custom_layers = keras.Sequential([
            keras.layers.Dense(1024, activation='selu',
                               kernel_initializer='lecun_normal'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation='selu',
                               kernel_initializer='lecun_normal'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='selu',
                               kernel_initializer='lecun_normal'),
            keras.layers.Dropout(0.1)
        ])
        self.output_layer = keras.layers.Dense(
            num_classes, activation='softmax', dtype='float32')

    def call(self, inputs, training=False):
        preprocessing = self.preprocess(inputs)
        augmentation = self.augmentation(preprocessing, training=training)
        base = self.base_model(augmentation, training=training)
        pooling = self.pooling(base)
        custom = self.custom_layers(pooling, training=training)
        outputs = self.output_layer(custom, training=training)
        return outputs




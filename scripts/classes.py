import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet152, EfficientNetB0, EfficientNetV2B0
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def get_base_model(model_name: str):
    '''Retrieves a pre-trained base model from TensorFlow's Keras applications.'''
    model_dict = {
        'ResNet50': ResNet50,
        'ResNet50V2': ResNet50V2,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152,
        'EfficientNetB0': EfficientNetB0,
        'EfficientNetV2B0': EfficientNetV2B0}
    if model_name in model_dict:
        return model_dict[model_name](include_top=False, weights='imagenet')
    raise ValueError(
        f'Invalid model name: {model_name}. Choose from {list(model_dict.keys())}.')


class AugmentLayer(keras.layers.Layer):
    def __init__(self, seed: int = 1113, factor: float = 0.2, dim: int = 256, crop_dim: int = 224):
        '''A custom Keras layer for applying data augmentation to input images.'''
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
    def __init__(self,
                 num_classes: int,
                 seed: int = 1113,
                 factor: float = 0.2,
                 dim: int = 256,
                 crop_dim: int = 224,
                 base_model: str = 'ResNet50',
                 frozen_layers: int = -50):
        '''A custom Keras model for classifying lichens.'''
        super().__init__()
        preprocess_map = {
            'ResNet50': resnet_preprocess,
            'ResNet50V2': resnet_preprocess,
            'ResNet101': resnet_preprocess,
            'ResNet152': resnet_preprocess,
            'EfficientNetB0': efficientnet_preprocess,
            'EfficientNetV2B0': efficientnet_preprocess
        }
        self.inputs = keras.Input(
            shape=(crop_dim, crop_dim, 3), dtype='float32')
        self.preprocess = preprocess_map[base_model]
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
        x = self.preprocess(inputs)
        x = self.augmentation(x, training=training)
        x = self.base_model(x, training=training)
        x = self.pooling(x)
        x = self.custom_layers(x, training=training)
        x = self.output_layer(x, training=training)
        return x
    
    def freeze_base_model(self):
        for layer in self.base_model.layers:
            layer.trainable = False
    def unfreeze_base_model(self, frozen_layers: int = None):
        for i, layer in enumerate(self.base_model.layers):
            if frozen_layers is not None and i < frozen_layers:
                layer.trainable = False
            else:
                layer.trainable = True

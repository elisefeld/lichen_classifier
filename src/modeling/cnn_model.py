import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet152, EfficientNetB0, EfficientNetV2B0
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras import optimizers

random.seed(1113)

class LichenClassifier(keras.Model):
    '''
    LichenClassifier is a custom Keras model designed for classifying lichens. It includes preprocessing, 
    data augmentation, a configurable base model, and custom dense layers for classification.

    Attributes:
        preprocessing_layer (keras.layers.Layer): A Lambda layer for applying the appropriate preprocessing 
            function based on the selected base model.
        augmentation (AugmentLayer): A custom data augmentation layer for applying transformations such as 
            rotation, contrast adjustment, and translation.
        base_model (keras.Model): The backbone model used for feature extraction. Can be one of several 
            pre-defined architectures (e.g., ResNet50, EfficientNetB0).
        pooling (keras.layers.GlobalMaxPooling2D): A global max pooling layer to reduce the spatial dimensions 
            of the feature maps.
        custom_layers (keras.Sequential): A sequential model containing custom dense layers for further 
            processing of the extracted features.
        output_layer (keras.layers.Dense): The final dense layer for classification, with softmax activation.

    Methods:
        __init__(self, num_classes, rotation_factor=0.2, contrast_factor=0.2, translation_factor=0.2, 
                 dim=256, crop_dim=224, base_model='ResNet50', frozen_layers=-50):
            Initializes the LichenClassifier with the specified parameters.

        call(self, inputs, training=False):
            Defines the forward pass of the model. Applies preprocessing, augmentation, feature extraction, 
            and classification to the input data.

        freeze_base_model(self):
            Freezes all layers in the base model, making them non-trainable.

        unfreeze_base_model(self, frozen_layers=None):
            Unfreezes layers in the base model. If `frozen_layers` is specified, only layers after the 
            specified index are made trainable.

    Args:
        num_classes (int): The number of output classes for classification.
        rotation_factor (float, optional): The maximum rotation angle for data augmentation. Defaults to 0.2.
        contrast_factor (float, optional): The maximum contrast adjustment factor for data augmentation. 
            Defaults to 0.2.
        translation_factor (float, optional): The maximum translation factor for data augmentation. 
            Defaults to 0.2.
        dim (int, optional): The input image dimension (height and width). Defaults to 256.
        crop_dim (int, optional): The dimension of the cropped image after augmentation. Defaults to 224.
        base_model (str, optional): The name of the base model architecture to use. Defaults to 'ResNet50'.
        frozen_layers (int, optional): The number of layers to freeze in the base model. Defaults to -50.
    '''
    def __init__(self,
                 num_classes: int,
                 rotation_factor: float = 0.2,
                 contrast_factor: float = 0.2,
                 translation_factor: float = 0.2,
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
        self.preprocessing_layer = keras.layers.Lambda(
            preprocess_map[base_model])
        self.augmentation = AugmentLayer(rotation_factor=rotation_factor, contrast_factor=contrast_factor,
                                         translation_factor=translation_factor, dim=dim, crop_dim=crop_dim)
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
        '''
        Executes the forward pass of the model.
        Args:
            inputs (tf.Tensor): Input tensor to the model.
            training (bool, optional): Indicates whether the model is in training mode. 
                Defaults to False.
        Returns:
            tf.Tensor: Output tensor after passing through all layers of the model.
        '''

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

def get_base_model(model_name: str):
    '''
    Retrieves a pre-trained base model from TensorFlow's Keras applications.

    Args:
        model_name (str): The name of the pre-trained model to retrieve. 
                          Supported models include:
                          - 'ResNet50'
                          - 'ResNet50V2'
                          - 'ResNet101'
                          - 'ResNet152'
                          - 'EfficientNetB0'
                          - 'EfficientNetV2B0'

    Returns:
        keras.Model: A Keras model instance with the specified architecture, 
                     pre-trained on the ImageNet dataset, and without the top classification layer.

    Raises:
        ValueError: If the provided model_name is not in the list of supported models.
    '''
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
    '''
    AugmentLayer is a custom Keras layer designed to apply data augmentation to input images.

    Attributes:
        rotation_factor (float): The range for random rotations represented as a fraction of 2pi. Default is 0.2 [-20% * 360, 20% * 360]
        #https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_rotation/
        contrast_factor (float): A positive float represented as fraction of value. Default is 0.2.
        #https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_contrast/
        translation_factor (tuple): A tuple of floats representing the height factor and width factor for translation. Default is (0.2, 0.2).
        #https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_translation/
        dim (int): The target dimension for resizing the input images. Default is 256.
        crop_dim (int): The target dimension for cropping the input images. Default is 224.

    Methods:
        call(inputs, training=False):
            Applies the augmentation pipeline to the input images during training. 
            If `training` is False, the inputs are returned without augmentation.

            Args:
                inputs: The input tensor containing image data.
                training (bool): A flag indicating whether the layer is in training mode. Default is False.

            Returns:
                Tensor: The augmented images if in training mode, otherwise the original inputs.
    '''

    def __init__(self,
                 rotation_factor: float = 0.2,
                 contrast_factor: float = 0.2,
                 translation_factor: tuple = [0.2, 0.2],
                 dim: int = 256,
                 crop_dim: int = 224):
        '''A custom Keras layer for applying data augmentation to input images.'''
        super().__init__()
        self.augment = keras.Sequential([
            keras.layers.Resizing(dim, dim, crop_to_aspect_ratio=True),
            keras.layers.RandomCrop(crop_dim, crop_dim),
            keras.layers.RandomFlip('horizontal'),
            keras.layers.RandomRotation(rotation_factor),
            keras.layers.RandomContrast(contrast_factor),
            keras.layers.RandomTranslation(translation_factor)
        ])

    def call(self, inputs, training=False):
        if training:
            return self.augment(inputs, training=training)
        return inputs


def get_optimizer(optimizer: str = 'SGD',
                  initial_learning_rate: float = 0.001,
                  use_schedule: bool = True,
                  schedule_type: str = 'exponential',
                  decay_steps: int = 1000,
                  decay_rate: float = 0.9,
                  staircase: bool = True,
                  **optimizer_kwargs):
    if use_schedule:
        if schedule_type == 'exponential':
            learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                  decay_steps=decay_steps,
                                                                  decay_rate=decay_rate,
                                                                  staircase=staircase)
        elif schedule_type == 'cosine':
            learning_rate = optimizers.schedules.CosineDecayRestarts(initial_learning_rate=initial_learning_rate,
                                                                     first_decay_steps=decay_steps,
                                                                     t_mul=1.0,  # can add params for these
                                                                     m_mul=1.0,
                                                                     alpha=0.0)
        else:
            raise ValueError(
                "Invalid schedule type. Choose from 'exponential' or 'cosine'.")
    else:
        learning_rate = initial_learning_rate

    if optimizer is not None and isinstance(optimizer, str):
        optimizer = optimizer.lower()

    if optimizer == 'sgd':
        return optimizers.SGD(learning_rate=learning_rate, **optimizer_kwargs)
    if optimizer == 'adam':
        return optimizers.Adam(learning_rate=learning_rate, **optimizer_kwargs)
    if optimizer == 'rmsprop':
        return optimizers.RMSprop(learning_rate=learning_rate, **optimizer_kwargs)
    if optimizer == 'adagrad':
        return optimizers.Adagrad(learning_rate=learning_rate, **optimizer_kwargs)
    if optimizer == 'adamax':
        return optimizers.Adamax(learning_rate=learning_rate, **optimizer_kwargs)
    if optimizer == 'nadam':
        return optimizers.Nadam(learning_rate=learning_rate, **optimizer_kwargs)
    raise ValueError(
        "Invalid optimizer name. Choose from 'sgd', 'adam', 'rmsprop', 'adagrad', 'adamax', or 'nadam' (case-insensitive).")

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    base_path: Path = Path('/Users/Elise/Code/esfeld/lichen_classifier')
    seed = 1113
    download: bool = False
    plot_imgs: bool = True
    plot_obs: bool = True
    mixed_precision: bool = True
    use_schedule: bool = True

    val_test_split: float = 0.15
    topk: int = 2

    dim: int = 256
    crop_dim: int = 224
    channels: int = 3

    rotation_factor: float = 0.2
    #transform_factor: float = 0.2
    contrast_factor: float = 0.2
    translation_factor: float = 0.2

    batch_size: int = 32
    epochs: int = 100
    patience: int = 10

    base_model: str = 'EfficientNetV2B0'  # ResNet50 or EfficientNetV2B0
    frozen_layers: int = -50
    optimizer: str = 'adam'
    learning_rate: float = 5e-4
    decay_steps: int = 30000
    decay_rate: float = 0.95
    schedule_type: str = 'cosine'
    smoothing: float = 0.05

    def __post_init__(self):
        self.data_dir = self.base_path / 'data'
        self.data_paths = [
            self.data_dir / 'raw' / 'obs_data_post17.csv',
            self.data_dir / 'raw' / 'obs_data_pre17.csv'
        ]
        self.counts_dir = self.data_dir / 'counts'
        self.location_dir = self.data_dir / 'location'
        self.image_dir = self.base_path / 'images'
        self.full_img_dir = self.image_dir / 'full'
        self.train_dir = self.image_dir / 'training' / 'train'
        self.val_dir = self.image_dir / 'training' / 'val'
        self.test_dir = self.image_dir / 'training' / 'test'
        self.results_dir = self.base_path / 'results'

    @property
    def input_shape(self):
        return (self.crop_dim, self.crop_dim, self.channels)

    @property
    def class_names(self):
        return [name.name for name in self.train_dir.iterdir() if name.is_dir()]
    
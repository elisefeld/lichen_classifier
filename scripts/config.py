from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    filter_list: List[str] = field(default_factory=list)
    topk: int = 5
    mixed_precision: bool = True
    seed: int = 1113
    val_test_split: float = 0.15
    download: bool = False
    base_path: Path = Path('/Users/Elise/Code/esfeld/lichen_classifier')
    dim: int = 256
    crop_dim: int = 224
    channels: int = 3
    transform_factor: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    base_model: str = 'EfficientNetV2B0'  # ResNet50 or EfficientNetV2B0
    frozen_layers: int = -50
    optimizer: str = 'adam'
    learning_rate: float = 5e-4
    decay_steps: int = 30000
    decay_rate: float = 0.95
    use_schedule: bool = True
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

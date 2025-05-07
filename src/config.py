from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    trial_num: int = 1
    base_path: Path = Path('/Users/Elise/Code/esfeld/lichen_classifier')
    seed: int = 1113
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
    contrast_factor: float = 0.2
    translation_factor: float = 0.2

    batch_size: int = 32
    epochs: int = 100
    patience: int = 5

    base_model: str = 'EfficientNetV2B0'  # ResNet50 or EfficientNetV2B0
    frozen_layers: int = 50
    optimizer: str = 'adam'
    coarse_learning_rate: float = 5e-4
    fine_learning_rate: float = 1e-5
    decay_steps: int = 30000
    decay_rate: float = 0.95
    schedule_type: str = 'exponential'
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
        self.training_history_dir = self.results_dir / 'training_history'
        self.confusion_matrix_dir = self.results_dir / 'confusion_matrix'
        self.class_metrics_dir = self.results_dir / 'class_metrics'
        self.histograms_dir = self.results_dir / 'histograms'
        self.time_plots_dir = self.results_dir / 'time_plots'
        self.location_plots_dir = self.results_dir / 'location_plots'

        dirs_to_create = [
            self.data_dir / 'raw',
            self.counts_dir,
            self.location_dir,
            self.full_img_dir,
            self.train_dir,
            self.val_dir,
            self.test_dir,
            self.training_history_dir,
            self.confusion_matrix_dir,
            self.class_metrics_dir,
            self.histograms_dir,
            self.time_plots_dir,
            self.location_plots_dir
        ]

        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def input_shape(self):
        return (self.crop_dim, self.crop_dim, self.channels)

    @property
    def train_classes(self):
        return [name.name for name in self.train_dir.iterdir() if name.is_dir()]
    
    @property
    def val_classes(self):
        return [name.name for name in self.val_dir.iterdir() if name.is_dir()]
    
    @property
    def test_classes(self):
        return [name.name for name in self.test_dir.iterdir() if name.is_dir()]
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.metrics import top_k_categorical_accuracy, CategoricalAccuracy

@dataclass
class Config:
    # Logging
    log_level: str = 'INFO'
    save_counts: bool = True
    save_location: bool = True

    # Data prep 
    download: bool = False
    val_test_split: float = 0.15
    plot_imgs: bool = True
    plot_obs: bool = True

    # General settings
    mixed_precision: bool = True
    trial_num: int = 2
    run_name: str = field(init=False)
    seed: int = 1113 # use fixed value for reproductibility, None for actually random number
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))

    # Paths
    base_path: Path = Path('/Users/Elise/Code/esfeld/lichen_classifier')
    data_dir: Path = field(init=False)
    image_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    # Training settings
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    optimizer: str = 'adam'
    topk: int = 2
    smoothing: float = 0.05
    fine_tune: bool = True
    frozen_layers: int = 50
    base_model: str = 'ResNet50'  # ResNet50 or EfficientNetV2B0 ***

    # Data augmentation
    dim: int = 256
    crop_dim: int = 224
    channels: int = 3
    rotation_factor: float = 0.2
    contrast_factor: float = 0.2
    translation_factor: float = 0.2

    # Scheduling 
    use_schedule: bool = True
    schedule_type: str = 'cosine'
    coarse_learning_rate: float = 1e-5
    fine_learning_rate: float = 1e-3
    
    # exponential decay
    decay_rate: float = 0.95
    decay_steps: int = 30000

    # cosine decay
    first_decay_steps: int = 10000
    t_mul: float = 2.0
    m_mul: float = 1.0
    alpha: float = 0.0

    def __post_init__(self):
        self.run_name = f"trial{self.trial_num}_{self.base_model}_{self.schedule_type}_{self.timestamp}"
        self.metrics = [keras.metrics.CategoricalAccuracy(name='categorical_accuracy'), keras.metrics.TopKCategoricalAccuracy(k=self.topk, name='top_k_categorical_accuracy')]

        # Main directories
        self.data_dir = self.base_path / 'data'
        self.image_dir = self.base_path / 'images'
        self.results_dir = self.base_path / 'results'
        self.model_dir = self.base_path / 'models'

        self.data_paths = [
            self.data_dir / 'raw' / 'obs_data_post17.csv',
            self.data_dir / 'raw' / 'obs_data_pre17.csv'
        ] #create a download from github for this or add the files

        # Subdirectories
        self.full_img_dir = self.image_dir / 'full'
        self.train_dir = self.image_dir / 'training' / 'train'
        self.val_dir = self.image_dir / 'training' / 'val'
        self.test_dir = self.image_dir / 'training' / 'test'

        self.counts_dir = self.data_dir / 'counts'
        self.location_dir = self.data_dir / 'location'
        
        self.training_history_dir = self.results_dir / 'training_history'
        self.confusion_matrix_dir = self.results_dir / 'confusion_matrix'
        self.class_metrics_dir = self.results_dir / 'class_metrics'
        self.histograms_dir = self.results_dir / 'histograms'
        self.time_plots_dir = self.results_dir / 'time_plots'
        self.location_plots_dir = self.results_dir / 'location_plots'
        

        dirs = [
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
            self.location_plots_dir,
            self.model_dir
        ]

        for d in dirs:
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
    
    def get_file_name(self, dir,  file_type: str, ext: str, test_type: str = None) -> Path:
        test_type = f"_{test_type}" if test_type else ""
        return dir / f"{file_type}_trial{self.trial_num}_{test_type}.{ext}"
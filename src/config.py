from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

@dataclass
class Config:
    # General settings
    trial_num: int = 5
    current_stage: int = field(default=1, init=False)
    num_stages: int = 2
    seed: int = 1113  # FIX use fixed value for reproductibility, None for actually random number
    base_path: Path = Path('/Users/Elise/Code/esfeld/lichen_classifier')
    run_name: str = field(init=False)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))

    # Logging
    log_level: str = 'INFO'
    save_counts: bool = True
    save_location: bool = True
    plot_imgs: bool = True
    plot_obs: bool = True

    # Data prep
    download: bool = False
    filter_download: bool = True
    val_test_split: float = 0.15
    filter_classes: bool = True
    filter_list: list[str] = field(default_factory=lambda: [
        'Xanthoria', 'Xanthomendoza', 'Vulpicida', 'Usnea', 'Umbilicaria', 'Teloschistes',
        'Rusavskia', 'Rhizoplaca', 'Punctelia', 'Porpidia', 'Platismatia', 'Pilophorus',
        'Physcia', 'Phaeophyscia', 'Peltigera', 'Parmotrema', 'Parmelia', 'Niebla',
        'Multiclavula', 'Lichenomphalia', 'Letharia', 'Lecanora', 'Lasallia', 'Icmadophila',
        'Heterodermia', 'Herpothallon', 'Graphis', 'Flavopunctelia', 'Evernia', 'Dimelaena',
        'Dibaeis', 'Candelaria', 'Arrhenia', 'Acarospora'])

    # Training settings
    mixed_precision: bool = True
    batch_size: int = 32
    epochs: int = 1  # 00
    patience: int = 10
    optimizer: str = 'adam'
    topk: int = 2
    smoothing: float = 0.05
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
    fine_learning_rate: float = 5e-6

    # exponential decay
    decay_rate: float = 0.95
    decay_steps: int = 30000

    # cosine decay
    first_decay_steps: int = 10000
    t_mul: float = 2.0
    m_mul: float = 1.0
    alpha: float = 0.0

    def __post_init__(self):
        self.run_name = f'trial{self.trial_num}_{self.base_model}_{self.schedule_type}_{self.timestamp}'

        # Main directories
        self.data_dir = self.base_path / 'data'
        self.image_dir = self.base_path / 'images'
        self.results_dir = self.base_path / 'results'

        # Subdirectories
        self.raw_dir = self.data_dir / 'raw'
        self.full_img_dir = self.image_dir / 'full'
        self.train_dir = self.image_dir / 'training' / 'train'
        self.val_dir = self.image_dir / 'training' / 'val'
        self.test_dir = self.image_dir / 'training' / 'test'
        self.EDA_dir = self.results_dir / 'EDA_results'
        self.run_dir = self.results_dir / self.run_name

        self.data_paths = [
            self.raw_dir / 'obs_data_post17.csv',
            # FIX create a download from github for this or add the files
            self.raw_dir / 'obs_data_pre17.csv']

        dirs = [
            self.raw_dir,
            self.full_img_dir,
            self.train_dir,
            self.val_dir,
            self.test_dir,
            self.EDA_dir,
            self.run_dir
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def metrics(self):
        return [
            keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            keras.metrics.TopKCategoricalAccuracy(
                k=self.topk, name='top_k_categorical_accuracy')
        ]

    def get_stage_lr(self, stage: int) -> float:
        return self.coarse_learning_rate * (self.fine_learning_rate / self.coarse_learning_rate) ** (stage / max(1, self.num_stages - 1))

    def get_stage_frozen_layers(self, stage: int) -> int:
        return int(self.frozen_layers * (1-(stage/max(1, self.num_stages - 1))))

    @property
    def input_shape(self):
        return (self.dim, self.dim, self.channels)

    @property
    def train_classes(self):
        return [name.name for name in self.train_dir.iterdir() if name.is_dir()]

    @property
    def val_classes(self):
        return [name.name for name in self.val_dir.iterdir() if name.is_dir()]

    @property
    def test_classes(self):
        return [name.name for name in self.test_dir.iterdir() if name.is_dir()]
    
    @property
    def num_classes(self) -> int:
        return len(np.unique(self.test_classes))
    
    @property
    def class_weights(self) -> dict[int, float]:
        class_weights_array = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(self.test_classes),
                y=self.test_classes)
        return {i: weight for i, weight in enumerate(class_weights_array)}

    # File names
    def get_file_name(self, file_type: str, ext: str, stage: int = None) -> Path:
        stage = self.current_stage if stage is None else stage
        return self.run_dir / f'{file_type}_trial{self.trial_num}_stage{self.current_stage}.{ext}'

    @property
    def json_results(self):
        return self.get_file_name('results', 'json')

    @property
    def history_csv(self):
        return self.get_file_name('training_history', 'csv')

    @property
    def history_plot(self):
        return self.get_file_name('training_history', 'png')

    @property
    def predictions_csv(self):
        return self.get_file_name('predictions', 'csv')

    @property
    def confusion_matrix_plot(self):
        return self.get_file_name('confusion_matrix', 'png')

    @property
    def class_metrics_csv(self):
        return self.get_file_name('class_metrics', 'csv')

    @property
    def class_metrics_plot(self):
        return self.get_file_name('class_metrics', 'png')

    @property
    def model_plot(self):
        return self.get_file_name('model', 'png')

    @property
    def model_checkpoint(self):
        return self.get_file_name(f'model_{self.base_model.lower()}_{self.optimizer.lower()}', 'keras')

    @property
    def prediction_visualization_plot(self):
        return self.get_file_name('prediction_examples', 'png')

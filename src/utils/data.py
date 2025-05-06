import logging
from pathlib import Path
import pandas as pd
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from config import Config 
cfg = Config()

# Set random seeds
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Functions ###
def load_and_clean_obs_data(paths: list[Path]) -> pd.DataFrame:
    # Read and concatenate CSV files
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    # Clean and transform data
    df['filename'] = df['uuid'].apply(lambda x: str(x) + '_full.jpg')
    df['large_image_url'] = df['image_url'].str.replace(
        'medium', 'large')
    df['genus'] = df['scientific_name'].str.split(' ').str[0]
    df['observed_on_dt'] = pd.to_datetime(df['observed_on'], errors='coerce')
    df['observed_on_month'] = df['observed_on_dt'].dt.month
    df['observed_on_day'] = df['observed_on_dt'].dt.day
    df['observed_on_year'] = df['observed_on_dt'].dt.year
    df['time_observed_at_dt'] = pd.to_datetime(
        df['time_observed_at'], errors='coerce')
    df['time_observed_at_hour'] = df['time_observed_at_dt'].dt.hour
    df['time_observed_at_minute'] = df['time_observed_at_dt'].dt.minute
    df['time_observed_at_second'] = df['time_observed_at_dt'].dt.second

    duplicate_uuids = df[df.duplicated('uuid', keep=False)]
    if not duplicate_uuids.empty:
        logger.info("Found duplicate UUIDs:\n%s",
                    duplicate_uuids.sort_values('uuid'))
    df = df.drop_duplicates(subset='uuid', keep='first')
    
    logger.info("Null values:\n%s", df.isnull().sum())
    # df = df.dropna(axis='rows', how='any')
    return df


def save_counts(df: pd.DataFrame, counts_dir: Path) -> None:
    counts_dir.mkdir(parents=True, exist_ok=True)
    for var in ['scientific_name', 'genus']:
        df[var].value_counts().to_csv(
            counts_dir / f'{var}_counts.csv', sep='\t')


def load_img_dataset(path: Path, batch_size: int = 32, dim: int = 224):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        shuffle=True,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(dim, dim),
        follow_links=True)
    return data


def get_class_info(ds: tf.data.Dataset):
    class_names = ds.class_names
    y_train = np.concatenate([np.argmax(y.numpy(), axis=1)
                              for _, y in ds])
    num_classes = len(np.unique(y_train))
    class_weights = dict(enumerate(
        class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train)))
    return class_names, num_classes, class_weights
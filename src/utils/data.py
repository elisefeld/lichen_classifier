import logging
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
import shutil
import time
import random
import requests
import cv2

from config import Config
cfg = Config()

# Set random seeds
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)
random.seed(cfg.seed)

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)

### Functions ###
def load_and_clean_obs_data(paths: list[Path] = cfg.data_paths) -> pd.DataFrame:
    # Read and concatenate CSV files
    logger.debug('Loading data from paths: %s', paths)
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    logger.info('Columns in Data: %s', df.columns.tolist())

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

    morphology_dict = {
        'Xanthomendoza': 'foliose',
        'Xanthoria': 'foliose',
        'Vulpicida': 'foliose',
        'Usnea': 'fruticose',
        'Umbilicaria': 'foliose',
        'Teloschistes': 'fruticose',
        'Rusavskia': 'foliose',
        'Rhizoplaca': 'foliose',
        'Punctelia': 'foliose',
        'Porpidia': 'crustose',
        'Platismatia': 'foliose',
        'Pilophorus': 'fruticose',
        'Physcia': 'foliose',
        'Parmotrema': 'foliose'
    }

    df['morphology'] = df['genus'].map(morphology_dict)

    # Remove duplicates
    duplicate_uuids = df[df.duplicated('uuid', keep=False)]
    if not duplicate_uuids.empty:
        logger.warning('Found duplicate UUIDs:\n%s',
                    duplicate_uuids.sort_values('uuid'))
    df = df.drop_duplicates(subset='uuid', keep='first')
    logger.warning('Null values:\n%s', df.isnull().sum())
    return df


def save_counts(df: pd.DataFrame, col: str = 'genus') -> None:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    df[col].value_counts().to_csv(cfg.EDA_dir/f'{col}_counts.csv')


def load_img_dataset(path: Path) -> tf.data.Dataset:
    data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        shuffle=True,
        labels='inferred',
        label_mode='categorical',
        batch_size=cfg.batch_size,
        image_size=(cfg.dim, cfg.dim),
        follow_links=True)
    logger.debug('Loaded dataset from %s', path)
    return data

def save_imgs(df: pd.DataFrame,
              to_filter: bool = cfg.filter_download,  
              filter_type: str = 'genus',
              filter_list: list = cfg.filter_list) -> list:
    
    if not cfg.download:
        logger.warning('Skipping image download as per configuration.')
        return []

    else: 
        logger.debug('Downloading images...')
        failed_uuids = []

        for _, row in df.iterrows():
            url, uuid, sci_name, genus = row['large_image_url'], row['uuid'], row['scientific_name'], row['genus']

            if to_filter:
                value = sci_name if filter_type == 'scientific_name' else genus
                if value not in filter_list:
                    logger.debug(f'Skipping {value} not in the list.')
                    continue

            folder_path = cfg.full_img_dir / genus / sci_name
            file_path = folder_path / f"{uuid}_{'full'}.jpg"
            folder_path.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                logger.debug('Image already exists: %s', file_path)
                continue

            try:
                time.sleep(1)
                res = requests.get(url, stream=True, timeout=60)
                res.raise_for_status()

                img_array = np.asarray(bytearray(res.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError(f'Failed to decode image {uuid}')

                cv2.imwrite(str(file_path), img)
                logger.debug('Image successfully downloaded:', file_path)

            except (requests.RequestException, ValueError) as e:
                logger.error(f'Failed to download/save image {uuid}: {e}')
                failed_uuids.append(uuid)

        return failed_uuids

def is_dir_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())

def train_test_split(source_dir: Path = cfg.full_img_dir,
                     dest_train_dir: Path = cfg.train_dir,
                     dest_test_dir: Path = cfg.test_dir,
                     dest_val_dir: Path = cfg.val_dir,
                     ratio: float = cfg.val_test_split,
                     overwrite: bool = False) -> None:

    if not overwrite and any(map(is_dir_nonempty, [dest_train_dir, dest_val_dir, dest_test_dir])):
            logger.warning("Split directories already contain files. Skipping train/test/val split.")
            return
    
    if overwrite:
        logger.info("Overwriting existing split directories.")
        for d in [dest_train_dir, dest_test_dir, dest_val_dir]:
            if d.exists():
                shutil.rmtree(d)

    for genus_dir in source_dir.iterdir():
        if not genus_dir.is_dir():
            continue
        for species_dir in genus_dir.iterdir():
            if not species_dir.is_dir():
                continue

            images = list(species_dir.glob('*.jpg')) + \
                list(species_dir.glob('*.png'))

            if len(images) < 3:
                print(
                    f'Not enough images in {species_dir.name} to split. Found {len(images)} images, but at least 3 are required.')
                continue

            num_test = max(1, int(len(images) * ratio))
            test_images = random.sample(images, num_test)
            remaining_images = [
                img for img in images if img not in test_images]

            num_val = max(1, int(len(remaining_images) * ratio))
            val_images = random.sample(remaining_images, num_val)
            train_images = [
                img for img in remaining_images if img not in val_images]

            splits = [('train', dest_train_dir, train_images),
                      ('test', dest_test_dir, test_images),
                      ('val', dest_val_dir, val_images)]

            for split_name, dest_root, imgs in splits:
                dest_species_dir = Path(dest_root) / \
                    genus_dir.name / species_dir.name
                dest_species_dir.mkdir(parents=True, exist_ok=True)

                for img in imgs:
                    shutil.copy(img, dest_species_dir)

                logger.info(
                    f'Copied {len(imgs)} images to {split_name} set: {genus_dir.name}/{species_dir.name}')

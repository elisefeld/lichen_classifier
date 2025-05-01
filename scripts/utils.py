from pathlib import Path
import logging
import shutil
import random
import time
import requests
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import optimizers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_counts(df: pd.DataFrame, counts_dir: Path):
    counts_dir.mkdir(parents=True, exist_ok=True)
    for var in ['scientific_name', 'genus']:
        df[var].value_counts().to_csv(
            counts_dir / f'{var}_counts.csv', sep='\t')


def save_imgs(df: pd.DataFrame,
              output_dir: Path,
              to_filter: bool,
              filter_type: str = 'scientific_name',
              filter_list: list = []) -> list:

    output_dir = Path(output_dir)
    failed_uuids = []

    for _, row in df.iterrows():
        url, uuid, sci_name, genus = row['large_image_url'], row['uuid'], row['scientific_name'], row['genus']

        if to_filter:
            value = sci_name if filter_type == 'scientific_name' else genus
            if value not in filter_list:
                print(f"Skipping {value} not in the list.")
                continue

        folder_path = output_dir / 'full' / genus / sci_name
        file_path = folder_path / f"{uuid}_{'full'}.jpg"
        folder_path.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            print('Image already exists:', file_path)
            continue

        try:
            time.sleep(1)
            res = requests.get(url, stream=True, timeout=60)
            res.raise_for_status()

            img_array = np.asarray(bytearray(res.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"Failed to decode image {uuid}")

            cv2.imwrite(str(file_path), img)
            print('Image successfully downloaded:', file_path)

        except (requests.RequestException, ValueError) as e:
            print(f"Failed to download/save image {uuid}: {e}")
            failed_uuids.append(uuid)

    return failed_uuids


def train_test_split(source_dir: Path,
                     dest_train_dir: Path,
                     dest_test_dir: Path,
                     dest_val_dir: Path,
                     ratio: float = 0.15,
                     seed: int = 1113):
    random.seed(seed)

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
                    f"Not enough images in {species_dir.name} to split. Found {len(images)} images, but at least 3 are required.")
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

                print(
                    f"Copied {len(imgs)} images to {split_name} set: {genus_dir.name}/{species_dir.name}")


def load_dataset(path: Path, batch_size: int = 32, dim: int = 224):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        shuffle=True,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(dim, dim),
        seed=1113,
        follow_links=True)
    return data


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
    else:
        raise ValueError(
            "Invalid optimizer name. Choose from 'SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adamax', or 'Nadam'.")


def load_with_metadata(
        image_dir: Path,
        location_csv: Path,
        batch_size: int = 32,
        dim: int = 224,
        shuffle: bool = True,
        seed: int = 1113):
    image_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(dim, dim),
        shuffle=shuffle,
        seed=seed,
    )

    filenames = sorted([p.name for p in image_dir.glob('*/*/*.jpg')])

    df = pd.read_csv(location_csv, sep='\t')  # or ',' depending on file
    df = df[df['filename'].isin(filenames)]
    df = df.sort_values('filename').reset_index(drop=True)

    latitudes = df['latitude'].astype('float32').values
    longitudes = df['longitude'].astype('float32').values
    locations = tf.convert_to_tensor(list(zip(latitudes, longitudes)), dtype=tf.float32)
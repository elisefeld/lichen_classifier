from pathlib import Path
import shutil
import time
import random
import requests
import cv2
import pandas as pd
import numpy as np

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

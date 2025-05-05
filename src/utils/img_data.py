import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

random.seed(1113)

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

"""
CSE 465 Assignment - Data preparation (run this once)

Downloads CIFAR-10, keeps the 2-class / 1000-sample subset, normalises and
flattens it, splits off the validation set, and saves everything to a single
.npz file. Every experiment script loads that file if it exists, so the
dataset is downloaded and processed only once - a failed experiment or a
typo costs no download time.

Run:
    python prepare_data.py

Output:
    data/cifar10_2class_1000.npz
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

# ----------------------------------------------------------------------
# Configuration (identical in every experiment file)
# ----------------------------------------------------------------------
CLASSES = (0, 1)        # 2 classes only -> 0 = airplane, 1 = automobile
N_TRAIN_SAMPLES = 1000  # 1000 samples only (before the train/validation split)
N_TEST_SAMPLES = 1000
SEED = 42

DATA_DIR = "data"
CACHE_PATH = os.path.join(DATA_DIR, "cifar10_2class_1000.npz")


def build_data():
    """CIFAR-10 -> normalised, flattened, 2-class, 1000-sample subset."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 3072)
    x_test = x_test.reshape(-1, 3072)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    def subset(x, y, n_samples):
        keep = np.isin(y, CLASSES)
        x, y = x[keep], y[keep]
        y = np.searchsorted(np.array(CLASSES), y)
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(x))[:n_samples]
        return x[idx], y[idx]

    x_pool, y_pool = subset(x_train, y_train, N_TRAIN_SAMPLES)
    x_test, y_test = subset(x_test, y_test, N_TEST_SAMPLES)

    x_train, x_val, y_train, y_val = train_test_split(
        x_pool, y_pool, test_size=0.2, random_state=42, stratify=y_pool
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def main():
    if os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH)
        print(f"Cache already exists: {CACHE_PATH}")
        print(f"train {d['x_train'].shape}  val {d['x_val'].shape}  test {d['x_test'].shape}")
        print("Delete the file if you want to rebuild it.")
        return

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = build_data()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )
    size_mb = os.path.getsize(CACHE_PATH) / 1e6
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")
    print(f"Class balance (train): {np.bincount(y_train)}")
    print(f"Saved {CACHE_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

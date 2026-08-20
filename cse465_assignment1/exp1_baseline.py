"""
CSE 465 Assignment - Experiment 1: Baseline Feedforward Neural Network

Architecture : 3072 -> Dense(256, relu) -> Dense(num_classes, softmax)
Optimizer    : SGD
Loss         : Sparse Categorical Crossentropy

Run:
    python exp1_baseline.py

Outputs:
    results/exp1_accuracy.png
    results/exp1_loss.png
    results/exp1_summary.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CLASSES = (0, 1)        # 2 classes only -> 0 = airplane, 1 = automobile
N_TRAIN_SAMPLES = 1000  # 1000 samples only (before the train/validation split)
N_TEST_SAMPLES = 1000   # test images taken from the CIFAR-10 test split
EPOCHS = 30
BATCH_SIZE = 64
SEED = 42

NUM_CLASSES = len(CLASSES)
RESULTS_DIR = "results"
DATA_DIR = "data"
# The prepared subset is cached here, so CIFAR-10 is downloaded and processed
# only once - every later run (or re-run after an error) loads this file.
CACHE_PATH = os.path.join(DATA_DIR, "cifar10_2class_1000.npz")


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def build_data():
    """CIFAR-10 -> normalised, flattened, 2-class, 1000-sample subset."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalise pixel values
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten 32 x 32 x 3 = 3072
    x_train = x_train.reshape(-1, 3072)
    x_test = x_test.reshape(-1, 3072)

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    def subset(x, y, n_samples):
        keep = np.isin(y, CLASSES)
        x, y = x[keep], y[keep]
        # Re-map original CIFAR labels to 0 .. NUM_CLASSES-1
        y = np.searchsorted(np.array(CLASSES), y)
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(x))[:n_samples]
        return x[idx], y[idx]

    x_pool, y_pool = subset(x_train, y_train, N_TRAIN_SAMPLES)
    x_test, y_test = subset(x_test, y_test, N_TEST_SAMPLES)

    # Validation set carved out of the training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_pool,
        y_pool,
        test_size=0.2,
        random_state=42,
        stratify=y_pool,
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_data():
    """Return the cached subset, or build it once and cache it."""
    if os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH)
        print(f"Loaded cached data from {CACHE_PATH}")
        return (
            (d["x_train"], d["y_train"]),
            (d["x_val"], d["y_val"]),
            (d["x_test"], d["y_test"]),
        )

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = build_data()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )
    print(f"Prepared data cached at {CACHE_PATH}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(3072,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="baseline_fnn",
    )
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_history(history, tag):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    h = history.history
    epochs = range(1, len(h["accuracy"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h["accuracy"], label="Training accuracy")
    plt.plot(epochs, h["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Experiment 1 - Training vs Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{tag}_accuracy.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h["loss"], label="Training loss")
    plt.plot(epochs, h["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Experiment 1 - Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{tag}_loss.png", dpi=150)
    plt.show()
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    keras.utils.set_random_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    plot_history(history, "exp1")

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("\n================ Experiment 1: Baseline ================")
    print(f"Training accuracy   : {train_acc:.4f}   (loss {train_loss:.4f})")
    print(f"Validation accuracy : {val_acc:.4f}   (loss {val_loss:.4f})")
    print(f"Test accuracy       : {test_acc:.4f}   (loss {test_loss:.4f})")
    print("========================================================")

    with open(f"{RESULTS_DIR}/exp1_summary.csv", "w") as f:
        f.write("model,train_acc,val_acc,test_acc,train_loss,val_loss,test_loss\n")
        f.write(
            f"baseline_256_relu_sgd,{train_acc:.4f},{val_acc:.4f},{test_acc:.4f},"
            f"{train_loss:.4f},{val_loss:.4f},{test_loss:.4f}\n"
        )


if __name__ == "__main__":
    main()

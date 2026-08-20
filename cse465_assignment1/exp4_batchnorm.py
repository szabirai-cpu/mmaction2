"""
CSE 465 Assignment - Experiment 4: Add Batch Normalization

Same architecture for both models (512 -> 256 -> softmax, ReLU, SGD):
    Model A : without Batch Normalization
    Model B : with Batch Normalization after every dense layer

Run:
    python exp4_batchnorm.py

Outputs:
    results/exp4_curves.png
    results/exp4_test_accuracy_bar.png
    results/exp4_summary.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CLASSES = (0, 1)        # 2 classes only -> 0 = airplane, 1 = automobile
N_TRAIN_SAMPLES = 1000  # 1000 samples only (before the train/validation split)
N_TEST_SAMPLES = 1000
EPOCHS = 30
BATCH_SIZE = 64
SEED = 42
HIDDEN_UNITS = [512, 256]   # best depth carried over from Experiment 3

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
def build_model(use_batchnorm, name):
    model = keras.Sequential(name=name)
    model.add(layers.Input(shape=(3072,)))
    for units in HIDDEN_UNITS:
        if use_batchnorm:
            # Dense -> BatchNorm -> activation
            model.add(layers.Dense(units, use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation("relu"))
        else:
            model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_curves(histories):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

    for col, (name, h) in enumerate(histories.items()):
        epochs = range(1, len(h["accuracy"]) + 1)

        axes[0, col].plot(epochs, h["accuracy"], label="Train")
        axes[0, col].plot(epochs, h["val_accuracy"], label="Validation")
        axes[0, col].set_title(f"{name} - Accuracy")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Accuracy")
        axes[0, col].legend()
        axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(epochs, h["loss"], label="Train")
        axes[1, col].plot(epochs, h["val_loss"], label="Validation")
        axes[1, col].set_title(f"{name} - Loss")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("Loss")
        axes[1, col].legend()
        axes[1, col].grid(alpha=0.3)

    fig.suptitle("Experiment 4 - Effect of Batch Normalization")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_curves.png", dpi=150)
    plt.show()
    plt.close(fig)


def plot_bar(results):
    labels = list(results)
    test_accs = [results[k]["test_acc"] for k in labels]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, test_accs, color=["gray", "seagreen"])
    for bar, acc in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2, acc, f"{acc:.3f}", ha="center", va="bottom"
        )
    plt.ylabel("Test accuracy")
    plt.title("Experiment 4 - Test Accuracy With and Without Batch Normalization")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp4_test_accuracy_bar.png", dpi=150)
    plt.show()
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    configs = {
        "Without BatchNorm": False,
        "With BatchNorm": True,
    }
    results, histories = {}, {}

    for name, use_bn in configs.items():
        keras.utils.set_random_seed(SEED)
        print(f"\n----- Training: {name} -----")

        model = build_model(use_bn, name.replace(" ", "_"))
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        histories[name] = history.history

        _, train_acc = model.evaluate(x_train, y_train, verbose=0)
        _, val_acc = model.evaluate(x_val, y_val, verbose=0)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)

        # Mean absolute epoch-to-epoch change of the validation loss:
        # a smaller value means a smoother / more stable loss curve.
        val_loss = np.array(history.history["val_loss"])
        smoothness = float(np.mean(np.abs(np.diff(val_loss))))

        results[name] = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "val_loss_jitter": smoothness,
        }

    plot_curves(histories)
    plot_bar(results)

    print("\n=============== Experiment 4: Batch Normalization ===============")
    print(f"{'Model':>19} {'Train acc':>11} {'Val acc':>9} {'Test acc':>10} {'Val-loss jitter':>17}")
    for name, r in results.items():
        print(
            f"{name:>19} {r['train_acc']:>11.4f} {r['val_acc']:>9.4f} "
            f"{r['test_acc']:>10.4f} {r['val_loss_jitter']:>17.4f}"
        )
    print("================================================================")

    with open(f"{RESULTS_DIR}/exp4_summary.csv", "w") as f:
        f.write("model,train_acc,val_acc,test_acc,val_loss_jitter\n")
        for name, r in results.items():
            f.write(
                f"{name},{r['train_acc']:.4f},{r['val_acc']:.4f},"
                f"{r['test_acc']:.4f},{r['val_loss_jitter']:.4f}\n"
            )


if __name__ == "__main__":
    main()

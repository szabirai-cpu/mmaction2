"""
CSE 465 Assignment - Experiment 5: Different Activation Functions

Architecture kept fixed (512 -> 256 -> softmax, BatchNorm, SGD); only the
hidden activation function changes:
    sigmoid, tanh, relu, leaky_relu  (+ optional elu, gelu, swish)

Run:
    python exp5_activations.py

Outputs:
    results/exp5_val_accuracy_curves.png
    results/exp5_test_accuracy_bar.png
    results/exp5_summary.csv
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
HIDDEN_UNITS = [512, 256]   # best architecture from Experiments 3 and 4
USE_BATCHNORM = True

ACTIVATIONS = ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu", "swish"]

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
def activation_layer(name):
    if name == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.01)
    return layers.Activation(name)


def build_model(activation):
    model = keras.Sequential(name=f"fnn_{activation}")
    model.add(layers.Input(shape=(3072,)))
    for units in HIDDEN_UNITS:
        model.add(layers.Dense(units, use_bias=not USE_BATCHNORM))
        if USE_BATCHNORM:
            model.add(layers.BatchNormalization())
        model.add(activation_layer(activation))
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
def plot_val_curves(histories):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for name, h in histories.items():
        plt.plot(range(1, len(h["val_accuracy"]) + 1), h["val_accuracy"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Experiment 5 - Validation Accuracy per Activation Function")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp5_val_accuracy_curves.png", dpi=150)
    plt.show()
    plt.close()


def plot_bar(results):
    labels = list(results)
    test_accs = [results[k]["test_acc"] for k in labels]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, test_accs, color="mediumpurple")
    for bar, acc in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2, acc, f"{acc:.3f}", ha="center", va="bottom"
        )
    plt.xlabel("Activation function")
    plt.ylabel("Test accuracy")
    plt.title("Experiment 5 - Test Accuracy vs Activation Function")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp5_test_accuracy_bar.png", dpi=150)
    plt.show()
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    results, histories = {}, {}

    for activation in ACTIVATIONS:
        keras.utils.set_random_seed(SEED)
        print(f"\n----- Training with activation: {activation} -----")

        model = build_model(activation)
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        histories[activation] = history.history

        _, train_acc = model.evaluate(x_train, y_train, verbose=0)
        _, val_acc = model.evaluate(x_val, y_val, verbose=0)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)

        # First epoch that reaches 90% of this model's best validation accuracy
        # -> a simple measure of how fast the activation trains.
        val_curve = np.array(history.history["val_accuracy"])
        target = 0.9 * val_curve.max()
        epochs_to_target = int(np.argmax(val_curve >= target)) + 1

        results[activation] = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "epochs_to_90pct": epochs_to_target,
        }

    plot_val_curves(histories)
    plot_bar(results)

    print("\n=============== Experiment 5: Activation Functions ===============")
    print(f"{'Activation':>12} {'Train acc':>11} {'Val acc':>9} {'Test acc':>10} {'Epochs to 90%':>15}")
    for name, r in results.items():
        print(
            f"{name:>12} {r['train_acc']:>11.4f} {r['val_acc']:>9.4f} "
            f"{r['test_acc']:>10.4f} {r['epochs_to_90pct']:>15}"
        )
    best = max(results, key=lambda k: results[k]["test_acc"])
    print(f"Best test accuracy: {best} ({results[best]['test_acc']:.4f})")
    print("=================================================================")

    with open(f"{RESULTS_DIR}/exp5_summary.csv", "w") as f:
        f.write("activation,train_acc,val_acc,test_acc,epochs_to_90pct_of_best_val\n")
        for name, r in results.items():
            f.write(
                f"{name},{r['train_acc']:.4f},{r['val_acc']:.4f},"
                f"{r['test_acc']:.4f},{r['epochs_to_90pct']}\n"
            )


if __name__ == "__main__":
    main()

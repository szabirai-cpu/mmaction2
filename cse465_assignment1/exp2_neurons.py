"""
CSE 465 Assignment - Experiment 2: Increase the Number of Neurons

Models (single hidden layer, ReLU, SGD, sparse categorical crossentropy):
    Model A : 256 neurons
    Model B : 512 neurons
    Model C : 1024 neurons

Run:
    python exp2_neurons.py

Outputs:
    results/exp2_curves.png
    results/exp2_test_accuracy_bar.png
    results/exp2_summary.csv
"""

import os

import matplotlib
matplotlib.use("Agg")
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
NEURON_CONFIGS = [256, 512, 1024]

NUM_CLASSES = len(CLASSES)
RESULTS_DIR = "results"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_data():
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


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(n_neurons):
    model = keras.Sequential(
        [
            layers.Input(shape=(3072,)),
            layers.Dense(n_neurons, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name=f"fnn_{n_neurons}",
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

    fig.suptitle("Experiment 2 - Effect of the Number of Neurons")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_curves.png", dpi=150)
    plt.close(fig)


def plot_bar(results):
    labels = [str(n) for n in results]
    test_accs = [results[n]["test_acc"] for n in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, test_accs, color="steelblue")
    for bar, acc in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )
    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("Test accuracy")
    plt.title("Experiment 2 - Test Accuracy vs Number of Neurons")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_test_accuracy_bar.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    results, histories = {}, {}

    for n_neurons in NEURON_CONFIGS:
        keras.utils.set_random_seed(SEED)   # same initial conditions for every model
        print(f"\n----- Training model with {n_neurons} neurons -----")

        model = build_model(n_neurons)
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        histories[f"{n_neurons} neurons"] = history.history

        _, train_acc = model.evaluate(x_train, y_train, verbose=0)
        _, val_acc = model.evaluate(x_val, y_val, verbose=0)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[n_neurons] = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "gap": train_acc - val_acc,
        }

    plot_curves(histories)
    plot_bar(results)

    print("\n=============== Experiment 2: Number of Neurons ===============")
    print(f"{'Neurons':>10} {'Train acc':>11} {'Val acc':>9} {'Test acc':>10} {'Train-Val gap':>14}")
    for n_neurons, r in results.items():
        print(
            f"{n_neurons:>10} {r['train_acc']:>11.4f} {r['val_acc']:>9.4f} "
            f"{r['test_acc']:>10.4f} {r['gap']:>14.4f}"
        )
    best = max(results, key=lambda k: results[k]["val_acc"])
    print(f"Best validation accuracy: {best} neurons ({results[best]['val_acc']:.4f})")
    print("==============================================================")

    with open(f"{RESULTS_DIR}/exp2_summary.csv", "w") as f:
        f.write("neurons,train_acc,val_acc,test_acc,train_val_gap\n")
        for n_neurons, r in results.items():
            f.write(
                f"{n_neurons},{r['train_acc']:.4f},{r['val_acc']:.4f},"
                f"{r['test_acc']:.4f},{r['gap']:.4f}\n"
            )


if __name__ == "__main__":
    main()

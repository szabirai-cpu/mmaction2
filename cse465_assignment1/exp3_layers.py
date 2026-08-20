"""
CSE 465 Assignment - Experiment 3: Increase the Number of Hidden Layers

Models (ReLU, SGD, sparse categorical crossentropy):
    Model A : 1 hidden layer  -> 512
    Model B : 2 hidden layers -> 512, 256
    Model C : 3 hidden layers -> 512, 256, 128

Run:
    python exp3_layers.py

Outputs:
    results/exp3_curves.png
    results/exp3_test_accuracy_bar.png
    results/exp3_summary.csv
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

# Hidden layer configurations: 1, 2 and 3 hidden layers
ARCHITECTURES = {
    "1 hidden layer": [512],
    "2 hidden layers": [512, 256],
    "3 hidden layers": [512, 256, 128],
}

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
def build_model(hidden_units, name):
    model = keras.Sequential(name=name.replace(" ", "_"))
    model.add(layers.Input(shape=(3072,)))
    for units in hidden_units:
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

    fig.suptitle("Experiment 3 - Effect of Network Depth")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_curves.png", dpi=150)
    plt.close(fig)


def plot_bar(results):
    labels = list(results)
    test_accs = [results[k]["test_acc"] for k in labels]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, test_accs, color="darkorange")
    for bar, acc in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2, acc, f"{acc:.3f}", ha="center", va="bottom"
        )
    plt.xlabel("Network depth")
    plt.ylabel("Test accuracy")
    plt.title("Experiment 3 - Test Accuracy vs Number of Hidden Layers")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp3_test_accuracy_bar.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    results, histories = {}, {}

    for name, hidden_units in ARCHITECTURES.items():
        keras.utils.set_random_seed(SEED)
        print(f"\n----- Training {name}: {hidden_units} -----")

        model = build_model(hidden_units, name)
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

        # Epoch at which validation accuracy peaked -> how fast the model overfits
        best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
        results[name] = {
            "layers": hidden_units,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "gap": train_acc - val_acc,
            "best_epoch": best_epoch,
        }

    plot_curves(histories)
    plot_bar(results)

    print("\n================== Experiment 3: Hidden Layers ==================")
    print(f"{'Model':>16} {'Train acc':>11} {'Val acc':>9} {'Test acc':>10} {'Gap':>8} {'Best epoch':>11}")
    for name, r in results.items():
        print(
            f"{name:>16} {r['train_acc']:>11.4f} {r['val_acc']:>9.4f} "
            f"{r['test_acc']:>10.4f} {r['gap']:>8.4f} {r['best_epoch']:>11}"
        )
    best = max(results, key=lambda k: results[k]["val_acc"])
    print(f"Best validation accuracy: {best} ({results[best]['val_acc']:.4f})")
    print("================================================================")

    with open(f"{RESULTS_DIR}/exp3_summary.csv", "w") as f:
        f.write("model,hidden_units,train_acc,val_acc,test_acc,train_val_gap,best_val_epoch\n")
        for name, r in results.items():
            f.write(
                f"{name},\"{r['layers']}\",{r['train_acc']:.4f},{r['val_acc']:.4f},"
                f"{r['test_acc']:.4f},{r['gap']:.4f},{r['best_epoch']}\n"
            )


if __name__ == "__main__":
    main()

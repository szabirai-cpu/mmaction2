"""
CSE 465 Assignment - Experiment 7: Different Loss Functions

Best architecture and optimizer so far (512 -> 256 -> softmax, BatchNorm,
ReLU, Adam) trained with:
    Sparse Categorical Crossentropy
    Categorical Crossentropy            (one-hot labels)
    Mean Squared Error                  (one-hot labels)
    Categorical Crossentropy + label smoothing   (optional)
    Categorical Focal Crossentropy               (optional)

Run:
    python exp7_losses.py

Outputs:
    results/exp7_val_accuracy_curves.png
    results/exp7_test_accuracy_bar.png
    results/exp7_summary.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

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
ACTIVATION = "relu"         # best activation from Experiment 5
LEARNING_RATE = 0.001       # best optimizer from Experiment 6 -> Adam

NUM_CLASSES = len(CLASSES)
RESULTS_DIR = "results"
DATA_DIR = "data"
# The prepared subset is cached here, so CIFAR-10 is downloaded and processed
# only once - every later run (or re-run after an error) loads this file.
CACHE_PATH = os.path.join(DATA_DIR, "cifar10_2class_1000.npz")

# name -> (loss object, one_hot_labels?)
def loss_functions():
    return {
        "Sparse CCE": (keras.losses.SparseCategoricalCrossentropy(), False),
        "Categorical CCE": (keras.losses.CategoricalCrossentropy(), True),
        "MSE": (keras.losses.MeanSquaredError(), True),
        "CCE + Label Smoothing": (
            keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            True,
        ),
        "Focal Loss": (keras.losses.CategoricalFocalCrossentropy(gamma=2.0), True),
    }


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
def build_model(loss, one_hot, name):
    model = keras.Sequential(name=name)
    model.add(layers.Input(shape=(3072,)))
    for units in HIDDEN_UNITS:
        model.add(layers.Dense(units, use_bias=not USE_BATCHNORM))
        if USE_BATCHNORM:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(ACTIVATION))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    # The metric is named "accuracy" in both cases so all runs are comparable.
    accuracy = (
        keras.metrics.CategoricalAccuracy(name="accuracy")
        if one_hot
        else keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=[accuracy],
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
    plt.title("Experiment 7 - Validation Accuracy per Loss Function")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp7_val_accuracy_curves.png", dpi=150)
    plt.show()
    plt.close()


def plot_bar(results):
    labels = list(results)
    test_accs = [results[k]["test_acc"] for k in labels]

    plt.figure(figsize=(9, 4))
    bars = plt.bar(labels, test_accs, color="indianred")
    for bar, acc in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2, acc, f"{acc:.3f}", ha="center", va="bottom"
        )
    plt.xlabel("Loss function")
    plt.ylabel("Test accuracy")
    plt.title("Experiment 7 - Test Accuracy vs Loss Function")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp7_test_accuracy_bar.png", dpi=150)
    plt.show()
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    results, histories = {}, {}

    for name, (loss, one_hot) in loss_functions().items():
        keras.utils.set_random_seed(SEED)
        print(f"\n----- Training with loss: {name} -----")

        y_tr = y_train_cat if one_hot else y_train
        y_va = y_val_cat if one_hot else y_val
        y_te = y_test_cat if one_hot else y_test

        model = build_model(loss, one_hot, name.replace(" ", "_").replace("+", "and"))
        history = model.fit(
            x_train,
            y_tr,
            validation_data=(x_val, y_va),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        histories[name] = history.history

        _, train_acc = model.evaluate(x_train, y_tr, verbose=0)
        _, val_acc = model.evaluate(x_val, y_va, verbose=0)
        _, test_acc = model.evaluate(x_test, y_te, verbose=0)

        # Training stability: mean absolute epoch-to-epoch change of the
        # validation accuracy (loss values are not comparable across
        # different loss functions, accuracy is).
        val_curve = np.array(history.history["val_accuracy"])
        stability = float(np.mean(np.abs(np.diff(val_curve))))

        results[name] = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "val_acc_jitter": stability,
        }

    plot_val_curves(histories)
    plot_bar(results)

    print("\n=================== Experiment 7: Loss Functions ===================")
    print(f"{'Loss':>23} {'Train acc':>11} {'Val acc':>9} {'Test acc':>10} {'Val-acc jitter':>16}")
    for name, r in results.items():
        print(
            f"{name:>23} {r['train_acc']:>11.4f} {r['val_acc']:>9.4f} "
            f"{r['test_acc']:>10.4f} {r['val_acc_jitter']:>16.4f}"
        )
    best = max(results, key=lambda k: results[k]["test_acc"])
    most_stable = min(results, key=lambda k: results[k]["val_acc_jitter"])
    print(f"Best test accuracy   : {best} ({results[best]['test_acc']:.4f})")
    print(f"Most stable training : {most_stable}")
    print("===================================================================")

    with open(f"{RESULTS_DIR}/exp7_summary.csv", "w") as f:
        f.write("loss,train_acc,val_acc,test_acc,val_acc_jitter\n")
        for name, r in results.items():
            f.write(
                f"{name},{r['train_acc']:.4f},{r['val_acc']:.4f},"
                f"{r['test_acc']:.4f},{r['val_acc_jitter']:.4f}\n"
            )


if __name__ == "__main__":
    main()

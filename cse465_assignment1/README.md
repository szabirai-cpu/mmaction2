# CSE 465 Assignment - CIFAR-10 with Feedforward Neural Networks

Each experiment from the assignment is a **separate, self-contained file** - no shared
imports between them, so any one file can be run (or pasted into a notebook) on its own.
The code only produces numbers, tables and plots; the written answers to the discussion
questions are left for the report.

## Files

| Experiment | Script | Kaggle notebook |
|---|---|---|
| 1. Baseline feedforward network | `exp1_baseline.py` | `kaggle_notebooks/exp1_baseline_kaggle.ipynb` |
| 2. Number of neurons (256 / 512 / 1024) | `exp2_neurons.py` | `kaggle_notebooks/exp2_neurons_kaggle.ipynb` |
| 3. Number of hidden layers (1 / 2 / 3) | `exp3_layers.py` | `kaggle_notebooks/exp3_layers_kaggle.ipynb` |
| 4. Batch normalization (with / without) | `exp4_batchnorm.py` | `kaggle_notebooks/exp4_batchnorm_kaggle.ipynb` |
| 5. Activation functions (sigmoid, tanh, relu, leaky relu, elu, gelu, swish) | `exp5_activations.py` | `kaggle_notebooks/exp5_activations_kaggle.ipynb` |
| 6. Optimizers (SGD, SGD+momentum, RMSprop, Adam, AdamW, Nadam) | `exp6_optimizers.py` | `kaggle_notebooks/exp6_optimizers_kaggle.ipynb` |
| 7. Loss functions (sparse CCE, CCE, MSE, label smoothing, focal) | `exp7_losses.py` | `kaggle_notebooks/exp7_losses_kaggle.ipynb` |

## Running on Kaggle

1. Create a new Kaggle notebook.
2. In the sidebar open **Settings** and switch **Internet** to **ON** - `cifar10.load_data()`
   downloads the dataset the first time it runs.
3. Upload the notebook for the experiment you want (`File -> Import Notebook`), or copy the
   contents of the matching `.py` file into a single code cell.
4. Run all. Plots appear inline and are also saved under `/kaggle/working/results/`, so they
   can be downloaded from the **Output** tab and dropped straight into the report.

CPU is enough - each experiment takes a couple of minutes because only 1000 training
samples are used.

## Running locally

```bash
pip install tensorflow scikit-learn matplotlib
python exp1_baseline.py     # then exp2 ... exp7
```

Plots and a `*_summary.csv` for each experiment are written to `results/`.

## Settings used by every experiment

These constants sit at the top of each file and are identical everywhere, so the seven
experiments stay comparable:

```python
CLASSES = (0, 1)        # 2 classes only: 0 = airplane, 1 = automobile
N_TRAIN_SAMPLES = 1000  # 1000 samples only, split 800 train / 200 validation
N_TEST_SAMPLES = 1000   # taken from the CIFAR-10 test split
EPOCHS = 30
BATCH_SIZE = 64
SEED = 42
```

Notes:

- Images are normalised to `[0, 1]` and flattened to `32 x 32 x 3 = 3072`, and only Dense
  layers are used - it is a feedforward network / MLP, as required.
- Because only 2 classes are used, the output layer has 2 softmax units and a random
  classifier scores about 0.50, not 0.10.
- The validation split is `train_test_split(..., test_size=0.2, random_state=42)` as given in
  the assignment, with `stratify` added so both classes stay balanced in the small subset.
- `keras.utils.set_random_seed(SEED)` is called before every model, so the models inside one
  experiment differ only by the setting being tested.
- Epochs: the assignment says 30 in the general instructions and 20 for Experiment 1. All
  files use 30 for consistency - change `EPOCHS` at the top of `exp1_baseline.py` to 20 if
  the 20-epoch version is preferred.
- Experiments 4-7 build on the previous result (2 hidden layers 512-256, then batch norm,
  then ReLU, then Adam). If your own runs point at a different winner, change the
  `HIDDEN_UNITS`, `USE_BATCHNORM`, `ACTIVATION` constants at the top of the later files.

## What each script prints and saves

- Training / validation accuracy and loss curves (`*_curves.png`, `*_accuracy.png`, `*_loss.png`)
- Bar chart of test accuracy for the compared settings (`*_test_accuracy_bar.png`)
- Validation-accuracy line plot per setting for Experiments 5-7 (`*_val_accuracy_curves.png`)
- A printed summary table plus `results/expN_summary.csv` with train / validation / test
  accuracy - the numbers to copy into the report tables.

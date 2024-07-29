# Baseprop
Baseline models for molecular property prediction. Currently, this package includes the GNN from “[Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)” and traditional MLP as model architectures. 

# Installing Baseprop
```python
conda create -n baseprop python=3.11
conda activate baseprop
git clone https://github.com/shihchengli/baseprop.git
cd baseprop
pip install -e .
```

# Using Baseprop
Baseprop can used either by CLI or as a Python module.

## CLI
Four different types of CLI are supported: `train`, `predict`, `hpopt`, and `nestedCV`. Below are some examples and descriptions of the arguments for each job. More details about other arguments can be found in the modules under the CLI folder.
### `train`: model training
```bash
baseprop train \
--data-path tests/data/freesolv.csv \
--task-type regression \
--output-dir train_example \
--smiles-columns smiles \
--target-columns freesolv \
--save-smiles-splits \
--split-type cv \
--num-folds 5 \
--molecule-featurizers morgan_binary
```
* `--data-path`: Path to an input CSV file containing SMILES and the associated target values.
* `--task-type`: Type of dataset. This determines the default loss function used during training. Defaults to regression.
* `--output-dir`: Directory where training outputs will be saved. Defaults to 'CURRENT_DIRECTORY/chemprop_training/STEM_OF_INPUT/TIME_STAMP'.
* `--smiles-columns`: The column names in the input CSV containing SMILES strings.
* `--target-columns`: Name of the columns containing target values.
* `--save-smiles-splits`: Save smiles for each train/val/test splits for prediction convenience later.
* `--split-type`: Method of splitting the data into train/val/test (case insensitive).
* `--num-folds`: Number of folds when performing cross validation.
* `--molecule-featurizers`: Method(s) of generating molecule features to use as extra descriptors.

### `predict`: model inference
```bash
baseprop predict \
--test-path freesolv.csv \
--preds-path train_example/fold_0/test_preds.csv \
--target-columns freesolv \
--model-path train_example/fold_0/model_0/best.pt \
--molecule-featurizers morgan_binary
```
* `--test-path`: Path to an input CSV file containing SMILES.
* `--preds-path`: Path to which predictions will be saved.
* `--model-path`: Location of checkpoint(s) or model file(s) to use for prediction.

### `hpopt`: hyperparameters optimization
```bash
baseprop cli \
--data-path freesolv.csv \
--task-type regression \
--smiles-columns smiles \
--target-columns freesolv \
--raytune-num-samples 5 \
--raytune-temp-dir $RAY_TEMP_DIR \
--raytune-num-cpus 40 \
--raytune-num-gpus 2 \
--raytune-max-concurrent-trials 2 \
--search-parameter-keywords depth ffn_num_layers hidden_channels ffn_hidden_dim dropout lr batch_size \
--hyperopt-random-state-seed 42 \
--hpopt-save-dir $results_dir
```
* `--raytune-num-samples`: Passed directly to Ray Tune TuneConfig to control number of trials to run.
* `--raytune-temp-dir`: Passed directly to Ray Tune init to control temporary director.
* `--raytune-num-cpus`: Passed directly to Ray Tune init to control number of CPUs to use.
* `--raytune-num-gpus`: Passed directly to Ray Tune init to control number of GPUs to use.
* `--raytune-max-concurrent-trials`: Passed directly to Ray Tune TuneConfig to control maximum concurrent trials.
* `--search-parameter-keywords`: The model parameters over which to search for an optimal hyperparameter configuration.
* `--hyperopt-random-state-seed`: Passed directly to HyperOptSearch to control random state seed.
* `--hpopt-save-dir`: Directory to save the hyperparameter optimization results.

### `nestedCV`: nested cross-validation (CV)
```bash
baseprop nestedCv \
--data-path freesolv.csv \
--task-type regression \
--smiles-columns smiles \
--target-columns freesolv \
--raytune-num-samples 20 \
--raytune-temp-dir $RAY_TEMP_DIR \
--raytune-num-cpus 40 \
--raytune-num-gpus 2 \
--raytune-max-concurrent-trials 2 \
--search-parameter-keywords depth ffn_num_layers hidden_channels ffn_hidden_dim dropout lr batch_size \
--hyperopt-random-state-seed 42 \
--hpopt-save-dir $results_dir \
--split-type cv \
--num-folds 5
```
**Note**: The number of CV folds in the outer and inner loops is the same as `--num-folds`.

# Python Module
Baseprop can also be used as a Python module to run baseline benchmarks or more complicated jobs. For example, there is a [notebook](https://github.com/shihchengli/baseprop/blob/main/examples/active_learning.ipynb) for active learning under the examples folder.

# Relationship to Chemprop
Baseprop is very similar to Chemprop, which uses a directed message passing (D-MPNN) neural network as the GNN model for chemical property prediction. Here, the GNN from “[Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)” is used as the baseline in the package. Additionally, the traditional MLP method can also be used with `--features-only` and `--molecule-featurizers` to only utilize fingerprints as input for the MLP. I ([@shihchengli](https://github.com/shihchengli)) am also a developer of Chemprop, so I adopted most of the code from Chemprop. This approach ensures a fair comparison between the model performance benchmark with D-MPNN and the other baselines implemented in this package.

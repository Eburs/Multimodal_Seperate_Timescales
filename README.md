# Code for "Learning Interpretable Hierarchical Dynamical Systems Models From Time Series Data" submitted at ICLR 2025

## Requirements

We include the `environment.yml` file to clone our conda environment. Simply run
```
conda env create -f environment.yml
```
to install and
```
conda activate hier-shPLRNN
```
to activate the environment.

## Usage

### Training
To start training a model, use the `main.py` file. Any hyper-parameters can be passed as command line arguments. Refer to the file, to see which hyper-parameters exist.
```
python main.py
```

Running multiple trainings, potentially in parallel, can be done via
```
python ubermain.py
```
any hyper-parameters must then be supplied in list format, see the file for more information. This also allows to do simple hyper-parameter grid searches.

### Evaluation
Trained models can be evaluated by
```
python main_eval.py --model_path <MODEL_PATH> --save_path <SAVE_PATH>
```
which will load all trained models in the `<MODEL_PATH>` directory and evaluate them in terms of state space divergence and average hellinger distance. The results will be saved to a file in `<SAVE_PATH>`.
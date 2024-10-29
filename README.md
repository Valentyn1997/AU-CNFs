AU-learner with conditional normalizing flows
==============================

Quantifying aleatoric uncertainty of the treatment effect with an AU-learner based on conditional normalizing flows (CNFs)

<img width="1666" alt="Preview 2024-10-29 19 00 37" src="https://github.com/user-attachments/assets/d5b67bee-574f-4fc3-9bab-976a80a5ec47">


The project is built with the following Python libraries:
1. [Pyro](https://pyro.ai/) - deep learning and probabilistic models (MDNs, NFs)
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking


## Setup

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000`

To access the MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to the local browser http://localhost:5000.

### Semi-synthetic datasets setup

Before running semi-synthetic experiments, place datasets in the corresponding folders:
- [IHDP100 dataset](https://www.fredjo.com/): ihdp_npci_1-100.test.npz and ihdp_npci_1-100.train.npz to `data/ihdp100/`

### Real-world case study 

We use multi-country data from [Banholzer et al. (2021)](https://doi.org/10.1371/journal.pone.0252827). Access data [here](https://github.com/nbanho/npi_effectiveness_first_wave/blob/master/data/data_preprocessed.csv).

## Experiments

The main training script is universal for different methods and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `config/` folder.

Generic script with logging and fixed random seed is the following:
```console
PYTHONPATH=.  python3 runnables/train.py +dataset=<dataset> +model=<model> exp.seed=10
```

### Models

One needs to choose a model and then fill in the specific hyperparameters (they are left blank in the configs):
- AU-CNFs (= AU-learner with CNFs, this paper): `+model=dr_cnfs` with two variants:
  - CRPS: `model.target_mode=cdf`
  - $W_2^2$: `model.target_mode=icdf`
- CA-CNFs (= CA-learner with CNFs, this paper): `+model=ca_cnfs` with two variants:
  - CRPS: `model.target_mode=cdf`
  - $W_2^2$: `model.target_mode=icdf`
- IPTW-CNF (= IPTW-learner with CNF, this paper): `+model=iptw_plugin_cnfs`
- Conditional Normalizing Flows (CNF, plug-in learner): `+model=plugin_cnfs`
- [Distributional Kernel Mean Embeddings](https://arxiv.org/pdf/1805.08845.pdf) (DKME): `+model=plugin_dkme`

Models already have the best hyperparameters saved (for each model and dataset), one can access them via: `+model/<dataset>_hparams=<model>` or `+model/<dataset>_hparams/<model>=<dataset_param>`. Hyperparameters for three variants of AU-CNFs, CA-CNFs, IPTW-CNF, and CNF are the same: `+model/<dataset>_hparams=plugin_cnfs`.

To perform a manual hyperparameter tuning, use the flags `model.tune_hparams=True`, and, then, see `model.hparams_grid`. 

### Datasets
One needs to specify a dataset/dataset generator (and some additional parameters, e.g. train size for the synthetic data `dataset.n_samples_train=1000`):
- Synthetic data (adapted from https://arxiv.org/abs/1810.02894): `+dataset=sine` with 3 settings:
  - Normal: `dataset.mode=normal`
  - Multi-modal: `dataset.mode=multimodal`
  - Exponential: `dataset.mode=exp`
- [IHDP](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162) dataset: `+dataset=ihdp` 
- [HC-MNIST](https://github.com/anndvision/quince/blob/main/quince/library/datasets/hcmnist.py) dataset: `+dataset=hcmnist`

### Examples

Example of running an experiment with our AU-CNFs (CRPS) on Synthetic data in the normal setting with $n_{\text{train}} = 100$ with 3 random seeds:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=sine +model=dr_cnfs +model/sine_hparams/plugin_cnfs_normal=\'100\' model.target_mode=cdf model.correction_coeff=0.25 exp.seed=10,101,1010
```

Example of tuning hyperparameters of the CNF based on HC-MNIST dataset:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=hcmnist +model=plugin_cnfs +model/hcmnist_hparams=plugin_cnfs exp.seed=10 model.tune_hparams=True
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

AU-learner with conditional normalizing flows
==============================

Quantifying aleatoric uncertainty of the treatment effect with an AU-learner based on conditional normalizing flows

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

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

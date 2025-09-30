# GAIGAR

`gaigar` is a Python package for **G**enomic **A**nalysis of **I**ntrogression at **G**enic and **A**llelic **R**esolution using machine learning. 
Currently, it supports two types of models:

- Logistic regression models
- U-Net models

`gaigar` uses established population genetic simulators like `msprime` for generating training and test data. 
It can be applied to detect introgressed fragments or alleles in genomes from various species.

## Requirements

`gaigar` works on UNIX/LINUX operating systems and tested with the following:

- Python 3.9
- Python packages:

    - demes=0.2.3
    - h5py=3.10.0
    - joblib=1.3.2
    - msprime=1.3.1
    - numpy=1.26.4
    - pandas=2.2.1
    - python=3.9.19
    - pyranges=0.0.129
    - pytest=8.1.1
    - scikit-allel=1.3.7
    - scikit-learn=1.4.1.post1
    - scipy=1.12.0
    - tskit=0.5.6
    - pyyaml=6.0.1
    - seriate==1.1.2
    - torch==2.2.0

## Installation

Users can install `gaigar` by using the following commands:

```
git clone https://github.com/xin-huang/gaigar
cd gaigar
mamba env create -f env.yaml
mamba activate gaigar
pip install .
```

Users first need to install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to create the virtual environment.

## Help

To get help information, users can use:

```         
gaigar -h
```

This will display information for three commands:

| Command | Description |
| - | - |
| lr | Use logistic regression models |
| unet | Use U-Net models |
| eval | Evaluate model performance |

If you need further help, such as reporting a bug or suggesting a feature, please open an [issue](https://github.com/xin-huang/gaigar/issues).

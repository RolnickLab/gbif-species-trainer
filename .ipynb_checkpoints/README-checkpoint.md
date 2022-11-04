# Species model trainer from GBIF 

This repository contains the code to download image and metadata for any species of interest from [GBIF](https://www.gbif.org/) (Global Biodiversity Information Facility) and train a deep learning model using the downloaded data.

## Setup python environment
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and prepare a python environment using the provided environment file:

Build a new conda environment
```bash
conda create -n gbif_species_trainer python=3.9
```

Install cuda toolkit and pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install additional libraries using pip:

```bash
pip install -r requirements.txt
```

Activate the conda environment:
```bash
conda activate gbif_species_trainer
```

## Download Data
The first step is to download data for species of interest. The instructions can be found in the `data_download` folder.

## Train Model
[TO BE UPDATED]

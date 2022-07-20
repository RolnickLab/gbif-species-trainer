## Model Training

[UNDER CONSTRUCTION]
The user needs to run the following scripts in a sequence to train the model:

1. `01-create_dataset_split.py`: Creates training, validation and testing splits of the data downloaded from GBIF.

```bash
python 01-create_dataset_split.py \
--data_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk-denmark/ \
--write_dir /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/ \
--train_ratio 0.75 \
--val_ratio 0.10 \
--test_ratio 0.15 \
--filename 01-uk-denmark
```

The description of the arguments to the script:
* `--data_dir`: Path to the root directory containing the GBIF data. **Required**.
* `--write_dir`: Path to the directory for saving the split files. **Required**.
* `--train_ratio`: Proportion of data for training. **Required**.
* `--val_ratio`: Proportion of data for validation. **Required**.
* `--test_ratio`: Proportion of data for testing. **Required**.
* `--filename`: Initial name for the split files. **Required**.

<br>
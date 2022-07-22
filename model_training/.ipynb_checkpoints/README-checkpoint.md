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


2. `02-calculate_taxa_statistics.py`: Calculates information and statistics regarding the taxonomy to be used for model training.

```bash
python 02-calculate_taxa_statistics.py \
--species_list /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark-Moth-List_11July2022.csv \
--write_dir /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/ \
--numeric_labels_filename uk-denmark_numeric_labels \
--taxon_hierarchy_filename uk-denmark_taxon_hierarchy \
--training_points_filename uk-denmark_count_training_points \
--train_split_file /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/01-uk-denmark-train-split.csv
```

The description of the arguments to the script:
* `--species_list`: Path to the species list. **Required**.
* `--write_dir`: Path to the directory for saving the information. **Required**.
* `--numeric_labels_filename`: Filename for numeric labels file. **Required**.
* `--taxon_hierarchy_filename`: Filename for taxon hierarchy file. **Required**.
* `--training_points_filename`: Filename for storing the count of training points. **Required**.
* `--train_split_file`: Path to the training split file. **Required**.

<br>


3. `03-create_webdataset.py`: Creates webdataset from raw image data. It needs to be run individually for each of the train, validation and test sets.

```bash
python 03-create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk-denmark/ \
--dataset_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/01-uk-denmark-train-split.csv \
--label_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/train/train-500-%06d.tar" 
```

The description of the arguments to the script:
* `--dataset_dir`: Path to the dataset directory containing the gbif data. **Required**.
* `--dataset_filepath`: Path to the csv file containing every data point information. **Required**.
* `--label_filepath`: File path containing numerical label information. **Required**.
* `--image_resize`: Resizing image factor (size x size) **Required**.
* `--webdataset_patern`: Path and format type to save the webdataset. It needs to be passed in double quotes. **Required**.
* `--max_shard_size`: The maximum shard size in bytes. Optional. **Default** is **10^8 (100 MB)**.
* `--random_seed`: Random seed for reproducible experiments. Optional. **Default** is **42**.
## Model Training

Run the following scripts in a sequence to train a model.

1. `01-create_dataset_split.py`: Creates training, validation, and testing splits of the data.

```bash
python 01-create_dataset_split.py \
--root_dir <full-path>/ \
--write_dir <full-path>/gbif_species_trainer/model_training/data/ \
--species_checklist <full-path>/[filename].csv \
--train_ratio 0.75 \
--val_ratio 0.10 \
--test_ratio 0.15 \
--filename 01-uk-denmark
```

The description of the arguments to the script:
* `--root_dir`: Path to the root directory containing the GBIF data. **Required**.
* `--write_dir`: Path to the directory for saving the split files. **Required**.
* `--species_checklist`: Path to the species checklist. **Required**.
* `--train_ratio`: Proportion of data for training. **Required**.
* `--val_ratio`: Proportion of data for validation. **Required**.
* `--test_ratio`: Proportion of data for testing. **Required**.
* `--filename`: Initial name for the split files. An example is shown above. **Required**.

<br>


2. `02-calculate_taxa_statistics.py`: Calculates information and statistics regarding the taxonomy and data to be used for model training.

```bash
python 02-calculate_taxa_statistics.py \
--species_checklist <full-path>/[filename].csv \
--write_dir <full-path>/gbif_species_trainer/model_training/data/ \
--numeric_labels_filename uk-denmark_numeric_labels_25Apr2023 \
--category_map_filename uk-denmark_category_map_25Apr2023 \
--taxon_hierarchy_filename uk-denmark_taxon_hierarchy_25Apr2023 \
--training_points_filename uk-denmark_count_training_points_25Apr2023 \
--train_split_file <full-path>/gbif_species_trainer/model_training/data/01-uk-denmark_train-split.csv
```

The description of the arguments to the script:
* `--species_checklist`: Path to the species checklist. **Required**.
* `--write_dir`: Path to the directory for saving the information. **Required**.
* `--numeric_labels_filename`: Filename for numeric labels file. An example is shown above. **Required**.
* `--category_map_filename`: Filename for the category map from integers to species names. An example is shown above. **Required**.
* `--taxon_hierarchy_filename`: Filename for taxon hierarchy file. An example is shown above. **Required**.
* `--training_points_filename`: Filename for storing the count of training points. An example is shown above. **Required**.
* `--train_split_file`: Path to the training split file. **Required**.

<br>


3. `03-create_webdataset.py`: Creates webdataset files from raw image data. It needs to be run separately for each of the train, validation, and test sets. Below is an example for training set but needs to be replicated for the other two.

```bash
python 03-create_webdataset.py \
--dataset_dir<full-path>/ \
--dataset_filepath <full-path>/gbif_species_trainer/model_training/data/01-uk-denmark_train-split.csv \
--label_filepath <full-path>/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels_25Apr2023.json \
--image_resize 500 \
--webdataset_patern "<full-path>/train/train-500-%06d.tar" 
```

The description of the arguments to the script:
* `--dataset_dir`: Path to the root directory containing the GBIF data. **Required**.
* `--dataset_filepath`: Path to the split file containing every data point information. **Required**.
* `--label_filepath`: File path containing numerical label information. **Required**.
* `--image_resize`: Resizing image factor (size x size) **Required**.
* `--webdataset_patern`: Path and format type to save the webdataset. It needs to be passed in double quotes. **Required**.
* `--max_shard_size`: The maximum shard size in bytes. Optional. **Default** is **10^8 (100 MB)**.
* `--random_seed`: Random seed for reproducible experiments. Optional. **Default** is **42**.

<br>

4. `04-train_model`: The main script for training the model.
```bash
python 04-train_model.py \
--train_webdataset_url "<full-path>/train-500-{000000..000900}.tar" \
--val_webdataset_url "<full-path>/val-500-{000000..000119}.tar" \
--test_webdataset_url "<full-path>test-500-{000000..000179}.tar" \
--config_file config/01-config_uk-denmark_efficientnet.json \
--dataloader_num_workers 6
```
The description of the arguments to the script:
* `--train_webdataset_url`: Path to the webdataset tar files for the training set. **Required**.
* `--val_webdataset_url`: Path to the webdataset tar files for the validation set. **Required**.
* `--test_webdataset_url`: Path to the webdataset tar files for the test set. **Required**.
* `--config_file`: Path to the configuration containing model training parameters. See example(s) in the `config` folder. **Required**.
* `--dataloader_num_workers`: Number of available CPUs. **Required**.
* `--random_seed`: Random seed for reproducible experiments. Optional. **Default** is **42**.

Note: For the webdataset url, the full count needs to be given in curly brackets.
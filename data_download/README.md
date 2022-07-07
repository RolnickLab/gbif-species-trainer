## Downloading Data from GBIF 

This folder contains the code to download data from GBIF. The user is **required to have a list of species names** and the code downloads media and meta data from GBIF for those corresponding species.


The following steps need to be executed in order:

### Step 1: Fetch unique keys
Step 1 involves fetching unique taxon keys for each species from GBIF Taxanomy backbone. 
```bash
python 01-fetch_taxon_keys.py \
--species_filepath example_species_lists/example2.csv \
--column_name beetles_species_names \
--output_filename beetles_30June2022
```
The description of the arguments to the script:
* `--species_filepath`: The user's list of species names. Example species lists are provided in the `example_species_lists` folder. **Required**.
* `--column_name`: The column name in the above csv file containing the species' names. **Required**.
* `--output_filename`: The output file name. **Required**.

### Step 2: Download data
Step 2 involves downloading data from GBIF. The description of the arguments to the script:

* `--write_directory`: Path to the folder to download the data. **Required**.
* `--species_list`: Path to the output csv file from `01-fetch_taxon_keys.py`. **Required**.
* `--max_images_per_species`: Maximum number of images to download for any species. Optional. **Default** is **500**.
* `--resume_session`: True/False; whether resuming a previously stopped downloading session. **Required**. **Default** is **False**.
* `--resume_session_index`: If `--resume_session` is `True`, row number in `--species_list` file from where downloading needs to resume (assuming 1-based indexing). Optional.

#### Fresh download
```bash
python 02a-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/gbif_data/ \
--species_list beetles_30June2022.csv \
--max_images_per_species 500 \
--resume_session False 
```

#### Resuming download
```bash
python 02a-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/gbif_data/ \
--species_list beetles_30June2022.csv \
--max_images_per_species 500 \
--resume_session True \
--resume_session_index 4
```



### Step 3: Create webdataset and remove corrupted images

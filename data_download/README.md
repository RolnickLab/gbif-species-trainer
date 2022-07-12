## Downloading Data from GBIF 

This folder contains the code to download data from GBIF. The user is **required to have a list of species names** and the code downloads media and meta data from GBIF for those corresponding species.


The following steps need to be executed in order:

### Step 1: Fetch unique keys
Step 1 involves fetching unique taxon keys for each species from GBIF Taxanomy backbone. 
```bash
python 01-fetch_taxon_keys.py \
--species_filepath example_species_lists/example2.csv \
--column_name beetles_species_names \
--output_filepath beetles_30June2022.csv
```
The description of the arguments to the script:
* `--species_filepath`: The user's list of species names. Example species lists are provided in the `example_species_lists` folder. **Required**.
* `--column_name`: The column name in the above csv file containing the species' names. **Required**.
* `--output_filepath`: The output file path with csv as extension. **Required**.

### Step 2: Download data
Step 2 involves downloading data from GBIF. The description of the arguments to the script:

* `--write_directory`: Path to the folder to download the data. **Required**.
* `--species_key_filepath`: Path to the output csv file from `01-fetch_taxon_keys.py`. **Required**.
* `--max_images_per_species`: Maximum number of images to download for any species. Optional. **Default** is **500**.
* `--resume_session`: `True` or `False`, whether resuming a previously stopped downloading session. Optional. **Default** is **False**.
* `--resume_session_index`: If `--resume_session` is `True`, row number in `--species_list` file from where downloading needs to resume (assuming 1-based indexing). Optional.

It is quite possible for the user to have a list of hundreds or thousands of species and maybe downloading half-a-million images. The downloading process is not too fast and can take days to complete in such cases. Hence, it is quite possible that the data needs to be fetched in parts. If the user is doing a fresh download, then command in [Fresh download](https://github.com/RolnickLab/gbif_species_trainer/tree/master/data_download#fresh-download) needs to be run. If the downloading is being resumed, the command in [Resuming download](https://github.com/RolnickLab/gbif_species_trainer/tree/master/data_download####Freshdownload) needs to be executed. 

If the user has already downloaded some data, there will be a `datacount.csv` file in the root directory `--write_directory`. This file shows the species for which the download is completed. When resuming, the index for the next species has to be found from `--species_key_filepath` file and given as input to `--resume_session_index` argument. For example, if download is completed till *Allotopus rosenbergi* for `beetles_30June2022.csv` file, then argument `--resume_session_index` will be 5 (for *Hexarthrius mandibularis*).

#### Fresh download
```bash
python 02a-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/gbif_data/ \
--species_key_filepath beetles_30June2022.csv \
--max_images_per_species 500 
```

#### Resuming download
```bash
python 02a-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/gbif_data/ \
--species_key_filepath beetles_30June2022.csv \
--max_images_per_species 500 \
--resume_session True \
--resume_session_index 5
```

There are two scripts for this step: `02a-fetch_gbif_moth_data.py` and `02b-fetch_gbif_other_data.py`. If the user needs to download data for moths and  species that have a similar life cycle (eggs-larvae-pupa-adult), `02a-fetch_gbif_moth_data.py` script ensures that images of only **adult** stage are downloaded. If this does not matter, for example the case of mammals, then `02b-fetch_gbif_other_data.py` should be used.


<!-- ### Step 3: Create webdataset and remove corrupted images
[TO BE UPDATED] -->
## Downloading data from GBIF 

This folder contains the code to download data from GBIF. The user is **required to have a list of species names** and the code downloads media and meta data from GBIF for those corresponding species. Examples species lists are provided in the `example_species_lists` folder.


The following steps need to be executed in order:

### Step 1: Fetch unique keys
Step 1 involves fetching unique taxon keys for each species from GBIF Taxanomy backbone. 
```bash
python 01-fetch_taxon_keys.py \
--species_filepath example_species_lists/example2.csv \
--column_name beetles_species_names \
--output_filename beetles_30June2022
```

### Step 2: Download data
```bash
python 02a-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/gbif_data/ \
--species_list beetles_30June2022.csv \
--max_images_per_species 500 \
--resume_session False 
```

```bash
python 02a-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/gbif_data/ \
--species_list beetles_30June2022.csv \
--max_images_per_species 500 \
--resume_session True \
--resume_session_index 4
```



### Step 3: Remove corrupted images

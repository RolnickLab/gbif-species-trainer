#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=4G                              # Ask for 4 GB of RAM
#SBATCH --output=create_webdataset.out


# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 4. Launch your script
python 03-create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk-denmark/ \
--dataset_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/01-uk-denmark-train-split.csv \
--label_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/train/train-500-%06d.tar" 

python 03-create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk-denmark/ \
--dataset_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/01-uk-denmark-val-split.csv \
--label_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/val/val-500-%06d.tar" \


python 03-create_webdataset.py \
--dataset_dir /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk/ \
--dataset_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/01-uk-denmark-test-split.csv \
--label_filepath /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/uk-denmark_numeric_labels.json \
--image_resize 500 \
--webdataset_patern "/home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/test/test-500-%06d.tar" \

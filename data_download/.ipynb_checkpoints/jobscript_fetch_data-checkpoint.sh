#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=2G                              # Ask for 2 GB of RAM
#SBATCH --output=fetch_data_output.out

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 4. Launch your script
python 02a-fetch_gbif_moth_data.py \
--write_directory /home/mila/a/aditya.jain/scratch/GBIF_Data/moths_uk-denmark/ \
--species_key_filepath /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark-Moth-List_11July2022.csv \
--max_images_per_species 500 \
--resume_session True 
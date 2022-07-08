#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=4G                              # Ask for 4 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Launch your script
python 01-fetch_taxon_keys.py \
--species_filepath /home/mila/a/aditya.jain/mothAI/species_lists/Denmark_original_May22.csv \
--column_name species_name \
--output_filepath /home/mila/a/aditya.jain/mothAI/species_lists/Denmark-Moth-List_08July2022.csv
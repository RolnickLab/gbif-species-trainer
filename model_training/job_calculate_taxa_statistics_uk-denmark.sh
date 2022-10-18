#!/bin/bash
#SBATCH --partition=main                      # Ask for main job
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=4G                              # Ask for 4 GB of RAM


# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 4. Launch your script
python 02-calculate_taxa_statistics.py \
--species_list /home/mila/a/aditya.jain/mothAI/species_lists/UK-Denmark_Moth-List_11July2022.csv \
--write_dir /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/ \
--numeric_labels_filename uk-denmark_numeric_labels \
--taxon_hierarchy_filename uk-denmark_taxon_hierarchy \
--training_points_filename uk-denmark_count_training_points \
--train_split_file /home/mila/a/aditya.jain/gbif_species_trainer/model_training/data/01-uk-denmark-train-split.csv



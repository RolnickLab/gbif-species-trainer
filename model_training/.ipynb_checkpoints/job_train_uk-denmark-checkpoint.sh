#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=6                     # Ask for 6 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=30G                             # Ask for 30 GB of RAM

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Copy your dataset on the compute node
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/train/train-500*.tar $SLURM_TMPDIR
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/val/val-500*.tar $SLURM_TMPDIR
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk-denmark/test/test-500*.tar $SLURM_TMPDIR


# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python 04-train_model.py \
--train_webdataset_url "$SLURM_TMPDIR/train-500-{000000..000697}.tar" \
--val_webdataset_url "$SLURM_TMPDIR/val-500-{000000..000092}.tar" \
--test_webdataset_url "$SLURM_TMPDIR/test-500-{000000..000139}.tar" \
--config_file config/01-config_uk-denmark.json \
--dataloader_num_workers 6


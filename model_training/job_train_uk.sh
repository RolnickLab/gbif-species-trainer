#!/bin/bash
#SBATCH --partition=main                      # Ask for main job
#SBATCH --cpus-per-task=8                     # Ask for 8 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=30G                             # Ask for 30 GB of RAM
#SBATCH --output=train_moth.out

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth

# 3. Copy your dataset on the compute node
rpath=home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk 

# 3. Copy your dataset on the compute node
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/train/train-500*.tar $SLURM_TMPDIR
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/val/val-500*.tar $SLURM_TMPDIR
cp /home/mila/a/aditya.jain/scratch/GBIF_Data/webdataset_moths_uk/test/test-500*.tar $SLURM_TMPDIR


# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python train_uk.py --data_path $SLURM_TMPDIR/$rpath --config_file config/01-config_uk.json

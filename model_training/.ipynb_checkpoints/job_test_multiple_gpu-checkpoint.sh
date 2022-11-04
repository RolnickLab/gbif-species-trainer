#!/bin/bash
#SBATCH --partition=main                # Ask for main job
#SBATCH --cpus-per-task=2              # Ask for 2 CPUs
#SBATCH --gres=gpu:v100:1            # Ask for 2 GPU
#SBATCH --mem=5G                       # Ask for 5 GB of RAM
#SBATCH --output=v100.out

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate milamoth_ai

# 4. Launch your job and look for the dataset into $SLURM_TMPDIR
python test_multiple_gpu.py 


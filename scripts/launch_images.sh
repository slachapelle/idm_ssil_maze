#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1

# Load modules or activate environments if needed
module load anaconda
conda activate idm_ssil_maze

NUM_SEEDS=5
MODEL=CNN

for MAZE in 10x10 20x20 50x50; do
    python learn_maze_img.py --output_folder exp/results_images --model $MODEL --maze $MAZE --num_goals 1 --num_seeds $NUM_SEEDS
done

NUM_SEEDS=5
MODEL=LINEAR_CNN

for MAZE in 10x10 20x20 50x50; do
    python learn_maze_img.py --output_folder exp/results_images --model $MODEL --maze $MAZE --num_goals 1 --num_seeds $NUM_SEEDS
done

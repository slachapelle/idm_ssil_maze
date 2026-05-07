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
NUM_INTER=1
MODEL=MLP

MAZE=10x10
for NUM_GOALS in 1 3 10; do
    python learn_maze.py --output_folder exp/results_goal --model $MODEL --num_inter $NUM_INTER --maze $MAZE --num_goals $NUM_GOALS --num_seeds $NUM_SEEDS
    python learn_maze.py --output_folder exp/results_goal --model $MODEL --num_inter $NUM_INTER --maze $MAZE --num_goals $NUM_GOALS --add_goal --num_seeds $NUM_SEEDS
done
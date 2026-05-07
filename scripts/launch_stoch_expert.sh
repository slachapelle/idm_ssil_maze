#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1

# Load modules or activate environments if needed
module load anaconda
conda activate idm_ssil_maze

NUM_SEEDS=10
MODEL=MLP
MAZE=20

for PROB in 0.5 0.9 1.0; do
    python learn_maze_stoch_expert.py --output_folder exp/results_stoch_expert --model $MODEL --env_size $MAZE --num_seeds $NUM_SEEDS --num_samples 1000 --expert_prob $PROB --max_iter 100000 #100000
done
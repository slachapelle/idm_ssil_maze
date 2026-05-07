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
MAZE=50x50
SAMPLES_MULT=4
ENV_VERSION=v2

MAX_ITER=200000
MODEL=MLP
for PROB in 0.25 0.5 1.0; do
   python learn_maze_stoch_env.py --output_folder exp/results_stoch_env --model $MODEL --maze $MAZE --env_version $ENV_VERSION --num_seeds $NUM_SEEDS --samples_mult $SAMPLES_MULT --env_prob $PROB --max_iter $MAX_ITER #100000 
done

MAX_ITER=200000
MODEL=LOGISTIC
for PROB in 0.25 0.5 1.0; do
   python learn_maze_stoch_env.py --output_folder exp/results_stoch_env --model $MODEL --maze $MAZE --env_version $ENV_VERSION --num_seeds $NUM_SEEDS --samples_mult $SAMPLES_MULT --env_prob $PROB --max_iter $MAX_ITER #100000 
done
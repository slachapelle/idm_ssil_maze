# On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning
**ICML 2026** 

*By Sacha Morin, Moonsub Byeon, Alexia Jolicoeur-Martineau, Sébastien Lachapelle*

Paper: https://arxiv.org/abs/2602.02762

This repo contains only the maze experiments. The code for the ProcGen and PushT/Libero experiments are also available:

- ProcGen: TODO
- PushT/Libero: TODO

## Installation
We recommend a conda environment based on python 3.8.2:
```
conda create -n idm_ssil_maze python=3.8.2
conda activate idm_ssil_maze
pip install -r requirements.txt
```
The code expects the directories `./exp` and `./data`.

## Data
The maze trajectories and optimal policies are generated using `mazelab` (package taken from https://github.com/zuoxingdong/mazelab/). You can generate the necessary data yourself using:

```
python generate_data.py
```
This should create folders for different maze sizes in `./data`.

## Running experiments
Everything will be saved under `./exp`

##### Figure 1 (Maze complexity)
```
source scripts/launch_states.sh
source scripts/launch_images.sh
```
##### Figure 2 (Goals)
```
source scripts/launch_goal.sh
```
##### Figure 3 (Stochastic expert)
```
source scripts/launch_stoch_expert.sh
```
##### Figure 7 (Stochastic environment)
```
source scripts/launch_stoch_env.sh
```
## Producing figures
```
python plot_final.py
```
The figures with appear in the `./exp` folders, next to the files produced by experiments.

## Reference
```
@inproceedings{
morin2026idm,
title={On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning},
author={Sacha Morin and Moonsub Byeon and Alexia Jolicoeur-Martineau and Sébastien Lachapelle},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=7hg1iTd34D}
}
```

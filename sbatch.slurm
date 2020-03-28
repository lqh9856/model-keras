#!/bin/bash

#SBATCH -J lqh-cross
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:
#SBATCH -o lqh-cross.out

python main.py





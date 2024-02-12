#!/bin/bash
#SBATCH --job-name=llama2_perp
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=254G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH --exclude=nikola-compute-[15-16]
#SBATCH -t 72:00:00
#SBATCH -o log_perp_noq.out                  # Name of stdout output     log file (%j expands to jobID)
#SBATCH -e log_perp_noq.err                  # Name of stderr output     log file (%j expands to jobID)
python3 perp.py | tee -a tee_perp_noq.out

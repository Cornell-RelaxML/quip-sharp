#!/bin/bash
#SBATCH --job-name=llama2
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=254G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH --exclude=nikola-compute-[15-18],badfellow,ellis-compute-[01-02]
#SBATCH -t 72:00:00
#SBATCH -o log_hess_70b.out                  # Name of stdout output     log file (%j expands to jobID)
#SBATCH -e log_hess_70b.err                  # Name of stderr output     log file (%j expands to jobID)
python3 hessian.py | tee -a hess_70b.out

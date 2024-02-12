#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=hessian-70b
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=750G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --constraint='v100'
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

python hessian_offline2.py --batch_size 2 --devset_size 4096 --ctx_size 4096 --save_path /home/at676/two_bit_quant/hessians/llama2_70b

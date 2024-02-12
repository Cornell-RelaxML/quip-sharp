#!/bin/bash
#SBATCH --partition=ellis,gpu
#SBATCH --job-name=main_check
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu-high'
#SBATCH --time=72:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

#python quantize_llama.py --save_path checkpoints/main_check
python hfize_llama.py --quantized_path checkpoints/main_check --hf_output_path hfized/main_check
python ppl_llama.py --hf_path hfized/main_check --dataset c4

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=half_int
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu-high'
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

python quantize_llama.py --save_path checkpoints/half_int --codebook half_integer_2bit
python hfize_llama.py --quantized_path checkpoints/half_int --hf_output_path hfized/half_int
python ppl_llama.py --hf_path hfized/half_int --dataset c4

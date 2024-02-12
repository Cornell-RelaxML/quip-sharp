#!/bin/bash
#SBATCH --partition=gpu,ellis
#SBATCH --job-name=d4_fast2
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu-high'
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

python quantize_llama.py --save_path checkpoints/d4_fast2 --codebook D4 
python hfize_llama.py --quantized_path checkpoints/d4_fast2 --hf_output_path hfized/d4_fast2
python ppl_llama.py --hf_path hfized/d4_fast2 --dataset c4

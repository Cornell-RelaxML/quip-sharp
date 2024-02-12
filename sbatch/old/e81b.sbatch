#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=e81b
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu-high'
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

CKPT=/share/desa/nfs02/quip_llama2/checkpoints/

python quantize_llama.py --save_path $CKPT/e81b --codebook E81B
python hfize_llama.py --quantized_path $CKPT/e81b --hf_output_path hfized/e81b
python ppl_llama.py --hf_path hfized/e81b --dataset c4

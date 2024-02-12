#!/bin/bash
#SBATCH --partition=gpu,ellis
#SBATCH --job-name=e8p12_3
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

CKPT=checkpoints

#python quantize_llama.py --save_path $CKPT/e8p12_3 --codebook E8P12
python hfize_llama.py --quantized_path $CKPT/e8p12_3 --hf_output_path hfized/e8p12_3
python ppl_llama.py --hf_path hfized/e8p12_3 --dataset c4

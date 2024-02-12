#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=d4_7b
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu-high'
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

CKPT=checkpoints

#python quantize_llama.py \
#       --save_path $CKPT/d4_7b \
#       --codebook D4 \
#       --base_model meta-llama/Llama-2-7b-hf \
#       --hessian_path /share/kuleshov/jc3464/quip/hessians/7b-chat-512dev-4096ctx \
#       --use_fp64

python hfize_llama.py --quantized_path $CKPT/d4_7b --hf_output_path hfized/d4_7b
python ppl_llama.py --hf_path hfized/d4_7b --dataset c4

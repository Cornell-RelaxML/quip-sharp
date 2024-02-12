#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=kmeans_8col
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu-high'
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

CKPT=/share/desa/nfs02/quip_llama2/checkpoints

#python quantize_llama.py --save_path $CKPT/kmeans_8col --codebook kmeans_8col
#python hfize_llama.py --quantized_path $CKPT/kmeans_8col --hf_output_path hfized/kmeans_8col
python ppl_llama.py --hf_path hfized/kmeans_8col --dataset c4

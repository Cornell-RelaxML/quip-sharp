#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=e8p_s090
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu-high'
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

CKPT=checkpoints

python quantize_llama.py --save_path $CKPT/e8p_s090 --codebook E8P12 --sigma_reg2 1e-2 --scale 0.90 --hessian_path /share/desa/nfs01/quip_llama2/hessians_llama2_70b_6144
python hfize_llama.py --quantized_path $CKPT/e8p_s090 --hf_output_path hfized/e8p_s090
python ppl_llama.py --hf_path hfized/e8p_s090 --dataset c4

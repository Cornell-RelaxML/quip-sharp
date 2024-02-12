#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=e8p_s105
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu-high'
#SBATCH --time=48:05:05
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

CKPT=checkpoints

python quantize_llama.py --save_path $CKPT/e8p_s105 --codebook E8P12 --sigma_reg2 1e-2 --scale 1.05 --hessian_path /share/desa/nfs01/quip_llama2/hessians_llama2_70b_6144
python hfize_llama.py --quantized_path $CKPT/e8p_s105 --hf_output_path hfized/e8p_s105
python ppl_llama.py --hf_path hfized/e8p_s105 --dataset c4

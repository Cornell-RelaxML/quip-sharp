#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=e8p_nofuse_ch
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

python quantize_llama_nofuse.py --save_path $CKPT/e8p_nofuse_ch --codebook E8P12 --sigma_reg2 1e-2 --scale 0.9 --hessian_path /share/desa/nfs01/quip_llama2/hessians_llama2_70b
python hfize_llama_nofuse.py --quantized_path $CKPT/e8p_nofuse_ch --hf_output_path hfized/e8p_nofuse_ch
python ppl_llama_nofuse.py --hf_path hfized/e8p_nofuse_ch --dataset c4

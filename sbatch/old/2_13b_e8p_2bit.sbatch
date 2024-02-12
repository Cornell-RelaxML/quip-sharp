#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=2_13b_e8p_2bit
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

'''
CKPT=checkpoints

python quantize_llama.py \
       --save_path $CKPT/2_13b_e8p_2bit \
       --codebook E8P12 \
       --sigma_reg2 1e-2 \
       --scale_override 0.9 \
       --base_model meta-llama/Llama-2-13b-hf \
       --hessian_path /share/desa/nfs01/quip_llama2/hessians_llama2_13b_6144

python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit --hf_output_path hfized/2_13b_e8p_2bit

'''

python ppl_llama.py --hf_path hfized/2_13b_e8p_2bit
python eval_llama.py --hf_path hfized/2_13b_e8p_2bit --batch_size 4 --tasks arc_challenge,arc_easy,boolq,piqa,winogrande

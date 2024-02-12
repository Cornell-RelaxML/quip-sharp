#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=hessian-7b_13b
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --constraint='v100'
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

rm /scratch/*.pt

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 4096 --save_path /share/desa/nfs01/quip_llama2/hessians_llama2_7b_6144 --scratch_path /scratch --base_model meta-llama/Llama-2-7b-hf

rm /scratch/*.pt

python hessian_offline.py --batch_size 2 --devset_size 6144 --ctx_size 4096 --save_path /share/desa/nfs01/quip_llama2/hessians_llama2_13b_6144 --scratch_path /scratch --base_model meta-llama/Llama-2-13b-hf


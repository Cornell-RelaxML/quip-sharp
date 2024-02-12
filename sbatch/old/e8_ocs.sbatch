#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=e8_ocs
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu-high'
#SBATCH --exclude='coecis-compute-03'
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

python quantize_llama.py --save_path checkpoints/e8_ocs --codebook E8 --outlier_channel_split
python hfize_llama.py --quantized_path checkpoints/e8_ocs --hf_output_path hfized/e8_ocs
python ppl_llama.py --hf_path hfized/e8_ocs_fast --dataset c4

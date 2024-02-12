#!/bin/bash
#SBATCH --partition=gpu,ellis
#SBATCH --job-name=ocs
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu-high|gpu-mid'
#SBATCH --time=72:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

#python quantize_llama.py --save_path checkpoints/ocs --outlier_channel_split
#python hfize_llama.py --quantized_path checkpoints/ocs --hf_output_path hfized/ocs
python c4_llama.py --hf_path hfized/ocs

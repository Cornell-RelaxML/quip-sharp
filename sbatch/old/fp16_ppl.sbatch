#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=fp16_ppl
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=at676@cornell.edu
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --constraint='gpu-high'
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

echo 'seqlen 2048'

python ppl_llama.py --hf_path meta-llama/Llama-2-70b-hf --seqlen 2048
python ppl_llama.py --hf_path meta-llama/Llama-2-13b-hf --seqlen 2048
python ppl_llama.py --hf_path meta-llama/Llama-2-7b-hf --seqlen 2048


echo 'seqlen 4096'

python ppl_llama.py --hf_path meta-llama/Llama-2-70b-hf --seqlen 4096
python ppl_llama.py --hf_path meta-llama/Llama-2-13b-hf --seqlen 4096
python ppl_llama.py --hf_path meta-llama/Llama-2-7b-hf --seqlen 4096



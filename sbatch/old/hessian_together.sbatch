#!/bin/bash
#SBATCH --job-name=hessian_llama_70b
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --err=slurm_out/%x_%j.err

cd /work/albert

python hessian_offline.py --batch_size 4 --devset_size 4096 --ctx_size 4096 --save_path /work/albert/two_bit_quant/hessians/llama2_70b --scratch_path /dev/shm

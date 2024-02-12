
#!/bin/bash

# output directory
# LEAVE COMMENTED OUT SO DON'T ACCIDENTALLY OVERWRITE
# dirname="hessians"
# logs_dirname="slurm_out/hessians"
# mkdir --parents $dirname
# mkdir --parents $logs_dirname

TIME="240"
CPU="8"
GPU="1"
# GPU="v100:1|a100:1"
GPUCONST="v100|a100"


TITLES=(
    "llama1-7b" "llama1-13b"\
    "llama1-30b" "llama1-65b"
    )
MODELS=(
    'decapoda-research/llama-7b-hf' 'decapoda-research/llama-13b-hf' \
    'decapoda-research/llama-30b-hf' 'decapoda-research/llama-65b-hf' 
    )
# GPUCONSTS=("gpu-mid" "gpu-mid" "gpu-high" "gpu-high")
MEMS=(
    "64G" "100G"\
    "160G" "200G"
    )
BSS=(
    "4" "4"\
    "4" "4"
    )
DEV="2048"
CTX="2048"


# main loop
for idx in "${!MODELS[@]}"
do
# save files
jobname="$Hessian_${TITLES[$idx]}"
echo $jobname
# slurm helper
slurm_helper="
#!/bin/bash
\n#SBATCH --job-name=${jobname}
\n#SBATCH -N 1
\n#SBATCH -c ${CPU}
\n#SBATCH --mail-type=FAIL
\n#SBATCH --mail-user=jc3464@cornell.edu
\n#SBATCH --partition=gpu
\n#SBATCH --gres=gpu:${GPU}
\n#SBATCH --constraint=\"${GPUCONST}\"
\n#SBATCH --mem=${MEMS[$idx]}
\n#SBATCH -t ${TIME}:00:00
\n#SBATCH -o ${logs_dirname}/${jobname}_%j.out
\n#SBATCH -e ${logs_dirname}/${jobname}_%j.err
\n\n
\nsource ~/.bashrc
\nsource ~/anaconda3/etc/profile.d/conda.sh
\nconda activate smoothquant
\n
\necho jobname: $jobname
\n\n
\necho '-------------------------------------'
\npython hessian_offline.py --devset_size ${DEV} --ctx_size ${CTX} --batch_size ${BSS[$idx]}
--base_model ${MODELS[$idx]} --save_path ${dirname}/${TITLES[idx]}-${DEV}dev-${CTX}ctx
"
# add slurm header to helper.sh
temp_file=$(mktemp)
echo -en $slurm_helper > $temp_file
echo $temp_file
# run on slurm
sbatch --requeue $temp_file

done
# \n#SBATCH --constraint=${GPUCONSTS[$idx]}
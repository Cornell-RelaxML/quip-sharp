
#!/bin/bash

# output directory
# LEAVE COMMENTED OUT SO DON'T ACCIDENTALLY OVERWRITE
dirname="checkpoints/llama1"
logs_dirname="slurm_out/llama1"
mkdir --parents $dirname
mkdir --parents $logs_dirname

MODELS=(
    # "7b"\
    "13b"\
    "30b"\
    "65b"
    )
MEMS=(
    # "32G"\
    "64G"\
    "160G"\
    "160G")
CONSTS=(
    # "gpu-mid"\
    "gpu-mid"\
    "gpu-mid"\
    "gpu-high"
    )
TIME="64"
CPU="8"
GPUS=(
    # "2"\
    "2"\
    "2"\
    "1"
    )

HESSIAN_PATHS=(
    # "hessians/llama1-7b-2048dev-2048ctx"\
    "hessians/llama1-13b-2048dev-2048ctx"\
    "hessians/llama1-30b-2048dev-2048ctx"\
    "hessians/llama1-65b-2048dev-2048ctx"
)
EXTRA_ARGS=(
    # "--lora_rank 128 --rescale_WH --outlier_channel_split --ocs_down_size 16384"\
    "--lora_rank 128 --rescale_WH --outlier_channel_split --ocs_down_size 16384"\
    "--lora_rank 128 --rescale_WH --outlier_channel_split --ocs_down_size 32768"\
    "--lora_rank 128 --rescale_WH --outlier_channel_split --ocs_down_size 32768"
    )
NAMES=(
    # "lora128_rescaleWH_ocs2-14"\
    "lora128_rescaleWH_ocs2-14"\
    "lora128_rescaleWH_ocs2-15"\
    "lora128_rescaleWH_ocs2-15"
    )


# main loop
for idx in "${!MODELS[@]}"
do
# save files
jobname="${MODELS[$idx]}_${NAMES[$idx]}"
# slurm helper
slurm_helper="
#!/bin/bash
\n#SBATCH --job-name=${jobname}
\n#SBATCH -N 1
\n#SBATCH -c ${CPU}
\n#SBATCH --mail-type=FAIL
\n#SBATCH --mail-user=jc3464@cornell.edu
\n#SBATCH --partition=gpu
\n#SBATCH --gres=gpu:${GPUS[$idx]}
\n#SBATCH --mem=${MEMS[$idx]}
\n#SBATCH --constraint=${CONSTS[$idx]}
\n#SBATCH -t ${TIME}:00:00
\n#SBATCH -o ${logs_dirname}/${jobname}_%j.out
\n#SBATCH -e ${logs_dirname}/${jobname}_%j.err
\n\n
\nsource ~/.bashrc
\nsource ~/anaconda3/etc/profile.d/conda.sh
\nconda activate smoothquant
\n
\necho jobname: $jobname
\necho extra args: ${EXTRA_ARGS[$idx]}
\n\n
\necho '-------------------------------------'
\npython quantize_llama.py --base_model decapoda-research/llama-${MODELS[$idx]}-hf 
${EXTRA_ARGS[$idx]} \
--save_path ${dirname}/${jobname} \
--hessian_path ${HESSIAN_PATHS[$idx]}
"
# add slurm header to helper.sh
temp_file=$(mktemp)
echo -en $slurm_helper > $temp_file
echo $temp_file
# run on slurm
# sbatch --requeue $temp_file

done

#!/bin/bash

# output directory
# LEAVE COMMENTED OUT SO DON'T ACCIDENTALLY OVERWRITE
dirname="checkpoints/llama1"
logs_dirname="slurm_out/llama1"
mkdir --parents $dirname
mkdir --parents $logs_dirname

# MODEL="7b-chat"
# HESSIAN_PATH="hessians/7b-chat-512dev-4096ctx"
# MODEL="13b-chat"
# HESSIAN_PATH="hessians/13b-chat-512dev-4096ctx"
MEM="32G"
CONST="gpu-mid"
TIME="64"
CPU="8"
GPU="2"

EXTRA_ARGS=("--lora_rank -1 --rescale_WH")
NAMES=("rescaleWHv2")


# main loop
for idx in "${!EXTRA_ARGS[@]}"
do
# save files
jobname="${MODEL}_${NAMES[$idx]}"
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
\n#SBATCH --mem=${MEM}
\n#SBATCH --constraint=${CONST}
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
\npython quantize_llama.py --base_model meta-llama/Llama-2-${MODEL}-hf
${EXTRA_ARGS[$idx]} \
--save_path ${dirname}/${jobname} \
--hessian_path $HESSIAN_PATH
"
# add slurm header to helper.sh
temp_file=$(mktemp)
echo -en $slurm_helper > $temp_file
echo $temp_file
# run on slurm
sbatch --requeue $temp_file

done
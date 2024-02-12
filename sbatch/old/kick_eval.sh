
#!/bin/bash

# output directory
# LEAVE COMMENTED OUT SO DON'T ACCIDENTALLY OVERWRITE
logs_dirname="slurm_out/eval_ablate1"
mkdir --parents $logs_dirname

MEM="32G"
CONST="gpu-mid"
TIME="64"
CPU="8"
GPU="1"
BS="4"

MODELS=(
    # "meta-llama/Llama-2-7b-chat-hf" #\
    "hfized/7b-chat_all" \
    "hfized/7b-chat_baseline_hada"\
    "hfized/7b-chat_lora"\
    "hfized/7b-chat_ocs"\
    "hfized/7b-chat_rescaleWH"\
    "hfized/7b-chat_rescaleWH_lora"
)
TASKS=("piqa" "winogrande" "arc_easy" "arc_challenge" "boolq")

# main loop
for mo_dir in "${MODELS[@]}"
do
for ta in "${TASKS[@]}"
do
# save file 
# mo_name=$(basename "$mo_dir")
mo_head=$(echo "$mo_dir" | cut -d / -f 1)
mo_name=$(echo "$mo_dir" | cut -d / -f 2)
jobname="${mo_head}_${mo_name}_${ta}"
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
\n\n
\necho '-------------------------------------'
\npython eval_llama.py --hf_path ${mo_dir} --tasks ${ta} --batch_size ${BS} --output_path ${logs_dirname}/${jobname}.json
"
# add slurm header to helper.sh
temp_file=$(mktemp)
echo -en $slurm_helper > $temp_file
echo $temp_file
# run on slurm
sbatch --requeue $temp_file

done
done
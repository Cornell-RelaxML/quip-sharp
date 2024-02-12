#!/bin/bash

source ~/miniconda3/bin/activate quipv2_mixtral

# TOKENIZERS_PARALLELISM=false python hessian_offline_mixtral.py \
#     --batch_size 2 --devset_size 64 --ctx_size 2048 --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --save_path /mnt/desa_data/hessians/mixtral_8x7b_RPv1_64dev2048ctx

# TOKENIZERS_PARALLELISM=false python hessian_offline_mixtral.py \
#     --batch_size 2 --devset_size 4096 --ctx_size 8192 --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --save_path /mnt/desa_data/hessians/mixtral_8x7b_RPv1_4096dev8192ctx

# TOKENIZERS_PARALLELISM=false python hessian_offline_mixtral.py \
#     --batch_size 1 --sample_proc 12 --devset_size 4096 --ctx_size 12288 \
#     --save_activations --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --save_path /mnt/desa_data/hessians/mixtral_8x7b_RPv1_4096dev12288ctx

# TOKENIZERS_PARALLELISM=false python hessian_offline_mixtral.py \
#     --batch_size 1 --sample_proc 12 --devset_size 4096 --ctx_size 8192 \
#     --save_activations --base_model mistralai/Mixtral-8x7B-Instruct-v0.1 \
#     --save_path /mnt/desa_data/hessians/mixtral_8x7b_instruct_RPv1_4096dev8192ctx

# TOKENIZERS_PARALLELISM=false python hessian_offline_mixtral.py \
#     --batch_size 1 --sample_proc 12 --devset_size 4096 --ctx_size 12288 \
#     --save_activations --base_model mistralai/Mixtral-8x7B-Instruct-v0.1 \
#     --save_path /mnt/desa_data/hessians/mixtral_8x7b_instruct_RPv1_4096dev12288ctx

TOKENIZERS_PARALLELISM=false python hessian_offline_mixtral.py \
    --batch_size 1 --sample_proc 12 --devset_size 4096 --ctx_size 12288  \
    --base_model mistralai/Mixtral-8x7B-v0.1 --dataset "togethercomputer/RedPajama-Data-V2" \
    --save_path /mnt/desa_data/hessians/mixtral_8x7b_RPv2_4096dev12288ctx
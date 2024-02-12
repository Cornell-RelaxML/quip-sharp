#!/bin/bash

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 2048 --base_model /work/desa_data/meta_llama1/huggingface_65B --save_path /work/desa_data/hessians/llama1_65b_6144

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 2048 --base_model /work/desa_data/meta_llama1/huggingface_30B --save_path /work/desa_data/hessians/llama1_30b_6144

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 2048 --base_model /work/desa_data/meta_llama1/huggingface_13B --save_path /work/desa_data/hessians/llama1_13b_6144

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 2048 --base_model /work/desa_data/meta_llama1/huggingface_7B --save_path /work/desa_data/hessians/llama1_7b_6144 

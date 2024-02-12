#!/bin/bash

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 4096 --base_model meta-llama/Llama-2-70b-chat-hf --save_path /work/desa_data/hessians/llama2_70b_chat_6144

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 4096 --base_model meta-llama/Llama-2-13b-chat-hf --save_path /work/desa_data/hessians/llama2_13b_chat_6144

python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 4096 --base_model meta-llama/Llama-2-7b-chat-hf --save_path /work/desa_data/hessians/llama2_7b_chat_6144


#!/bin/bash

python hessian_offline.py --batch_size 2 --devset_size 4096 --ctx_size 8192 --base_model mistralai/Mistral-7B-v0.1 --save_path /work/desa_data/hessians/mistral_7b_4096

python hessian_offline.py --batch_size 2 --devset_size 4096 --ctx_size 8192 --base_model teknium/OpenHermes-2.5-Mistral-7B --save_path /work/desa_data/hessians/openhermes_7b_4096




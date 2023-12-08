#!/bin/bash

LOG=/mnt/desa_data/logs

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --model-dir /mnt/desa_data/q2k/llama-v1-7b-q2k.gguf --use_flash_attention_2 --n-gpu-layers 10000 >> $LOG/1_7b_q2k 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --model-dir /mnt/desa_data/q2k/llama-v1-13b-q2k.gguf --use_flash_attention_2 --n-gpu-layers 10000 >> $LOG/1_13b_q2k 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --model-dir /mnt/desa_data/q2k/llama-v1-30b-q2k.gguf --use_flash_attention_2 --n-gpu-layers 10000 >> $LOG/1_30b_q2k 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --model-dir /mnt/desa_data/q2k/llama-v1-65b-q2k.gguf --use_flash_attention_2 --n-gpu-layers 10000 >> $LOG/1_65b_q2k 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --model-dir /mnt/desa_data/q2k/llama-v2-7b-q2k.gguf --use_flash_attention_2 --n-gpu-layers 10000 >> $LOG/2_7b_q2k 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --model-dir /mnt/desa_data/q2k/llama-v2-13b-q2k.gguf --use_flash_attention_2 --n-gpu-layers 10000 >> $LOG/2_13b_q2k 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --model-dir /mnt/desa_data/q2k/llama-v2-70b-q2k.gguf --use_flash_attention_2 --n-gpu-layers 10000 >> $LOG/2_70b_q2k 2>&1 &

wait

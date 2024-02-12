#!/bin/bash

CKPT=/mnt/desa_data/checkpoints/finetune_albert
HF=/mnt/desa_data/hfized/finetune_albert
HESS=/mnt/desa_data/hessians
LOG=/mnt/desa_data/logs/finetune_albert

mkdir $CKPT
mkdir $HF
mkdir $LOG

CUDA_VISIBLE_DEVICES=0,1 python quantize_llama_finetune.py --save_path $CKPT/2_70b_2bit  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144  --devset_size 768 --ddp_port 12345 >> $LOG/2_70b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 python quantize_llama_finetune.py --save_path $CKPT/2_70b_3bit  --codebook E8P12RVQ3B --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144  --devset_size 768 --ddp_port 12346 >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 python quantize_llama_finetune.py --save_path $CKPT/1_65b_2bit  --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-65b-hf --hessian_path $HESS/llama1_65b_6144  --devset_size 768 --ddp_port 12347 >> $LOG/1_65b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 python quantize_llama_finetune.py --save_path $CKPT/1_65b_3bit  --codebook E8P12RVQ3B --base_model relaxml/Llama-1-65b-hf --hessian_path $HESS/llama1_65b_6144  --devset_size 768 --ddp_port 12348 >> $LOG/1_65b_3bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python quantize_llama_finetune.py --save_path $CKPT/2_13b_2bit  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144  --devset_size 768 --ddp_port 12345 >> $LOG/2_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python quantize_llama_finetune.py --save_path $CKPT/2_13b_3bit  --codebook E8P12RVQ3B --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144  --devset_size 768 --ddp_port 12346 >> $LOG/2_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python quantize_llama_finetune.py --save_path $CKPT/1_13b_2bit  --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-13b-hf --hessian_path $HESS/llama1_13b_6144  --devset_size 768 --ddp_port 12347 >> $LOG/1_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python quantize_llama_finetune.py --save_path $CKPT/1_13b_3bit  --codebook E8P12RVQ3B --base_model relaxml/Llama-1-13b-hf --hessian_path $HESS/llama1_13b_6144  --devset_size 768 --ddp_port 12348 >> $LOG/1_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 python quantize_llama_finetune.py --save_path $CKPT/1_30b_2bit  --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-30b-hf --hessian_path $HESS/llama1_30b_6144  --devset_size 768 --ddp_port 12349 >> $LOG/1_30b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 python quantize_llama_finetune.py --save_path $CKPT/1_30b_3bit  --codebook E8P12RVQ3B --base_model relaxml/Llama-1-30b-hf --hessian_path $HESS/llama1_30b_6144  --devset_size 768 --ddp_port 12350 >> $LOG/1_30b_3bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python quantize_llama_finetune.py --save_path $CKPT/2_7b_2bit  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144  --devset_size 768 --ddp_port 12345 >> $LOG/2_7b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python quantize_llama_finetune.py --save_path $CKPT/2_7b_3bit  --codebook E8P12RVQ3B --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144  --devset_size 768 --ddp_port 12346 >> $LOG/2_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python quantize_llama_finetune.py --save_path $CKPT/1_7b_2bit  --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-7b-hf --hessian_path $HESS/llama1_7b_6144  --devset_size 768 --ddp_port 12347 >> $LOG/1_7b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python quantize_llama_finetune.py --save_path $CKPT/1_7b_3bit  --codebook E8P12RVQ3B --base_model relaxml/Llama-1-7b-hf --hessian_path $HESS/llama1_7b_6144  --devset_size 768 --ddp_port 12348 >> $LOG/1_7b_3bit 2>&1 &

CUDA_VISIBLE_DEVICES=4 python quantize_llama_finetune.py --save_path $CKPT/2_7b_4bit  --codebook E8P12RVQ4B --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144  --devset_size 768 --ddp_port 12349 >> $LOG/2_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python quantize_llama_finetune.py --save_path $CKPT/1_7b_4bit  --codebook E8P12RVQ4B --base_model relaxml/Llama-1-7b-hf --hessian_path $HESS/llama1_7b_6144  --devset_size 768 --ddp_port 12350 >> $LOG/1_7b_4bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0,1 python quantize_llama_finetune.py --save_path $CKPT/1_30b_4bit  --codebook E8P12RVQ4B --base_model relaxml/Llama-1-30b-hf --hessian_path $HESS/llama1_30b_6144  --devset_size 768 --ddp_port 12345 >> $LOG/1_30b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 python quantize_llama_finetune.py --save_path $CKPT/2_70b_4bit  --codebook E8P12RVQ4B --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144  --devset_size 768 --ddp_port 12346 >> $LOG/2_70b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 python quantize_llama_finetune.py --save_path $CKPT/1_65b_4bit  --codebook E8P12RVQ4B --base_model relaxml/Llama-1-65b-hf --hessian_path $HESS/llama1_65b_6144  --devset_size 768 --ddp_port 12347 >> $LOG/1_65b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python quantize_llama_finetune.py --save_path $CKPT/2_13b_4bit  --codebook E8P12RVQ4B --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144  --devset_size 768 --ddp_port 12351 >> $LOG/2_13b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python quantize_llama_finetune.py --save_path $CKPT/1_13b_4bit  --codebook E8P12RVQ4B --base_model relaxml/Llama-1-13b-hf --hessian_path $HESS/llama1_13b_6144  --devset_size 768 --ddp_port 12352 >> $LOG/1_13b_4bit 2>&1 &
wait

#!/bin/bash
set -euxo pipefail


CKPT=/share/desa/nfs01/qs234/quip-sharp/checkpoints
HF=/share/desa/nfs01/qs234/quip-sharp/hfized
HESS=/share/desa/nfs01/qs234/huggingface/hub/models--relaxml--Hessians-Llama-2-7b-6144/snapshots/cafc59c036c6416ec2a9d5790752bec51297c197/
LOG=/share/desa/nfs01/qs234/quip-sharp/logs


mkdir -p $CKPT
mkdir -p $HF
mkdir -p $LOG


# quantize with finetuning
python3 \
    -m quip_sharp.quantize_llama.quantize_finetune_llama \
    --save_path $CKPT/2_7b_2bit \
    --codebook E8P12 \
    --scale_override 0.9 \
    --base_model meta-llama/Llama-2-7b-hf \
    --hessian_path $HESS \
    --devset_size 384 \
    --ft_valid_size 128 \
    --ft_epochs 8 \
    2>&1 \
    | tee -a $LOG/2_7b_2bit


# convert model to hf format for end to end fine tuning
CUDA_VISIBLE_DEVICES=0 python3 \
    -m quip_sharp.quantize_llama.hfize_llama \
    --quantized_path $CKPT/2_7b_2bit \
    --hf_output_path $HF/2_7b_2bit \
    2>&1 \
    | tee -a $LOG/2_7b_2bit


# end to end fine tuning
# python3 \
#     -m quip_sharp.quantize_llama.finetune_e2e_llama \
#     --base_model meta-llama/Llama-2-7b-hf \
#     --hf_path $HF/2_7b_2bit \
#     --devset_size 384 \
#     --ft_valid_size 128 \
#     --ft_epochs 8 \
#     --ft_bs 1 \
#     --ctx_size 4096 \
#     --ft_update_freq 2 \
#     --ft_train_mode \
#     --ckpt_path $CKPT/2_7b_2bit \
#     2>&1 \
#     | tee -a $LOG/2_7b_2bit


# eval
CUDA_VISIBLE_DEVICES=0 python3 \
    -m quip_sharp.quantize_llama.hfize_llama \
    --quantized_path $CKPT/2_7b_2bit \
    --hf_output_path $HF/2_7b_2bit \
    2>&1 \
    | tee -a $LOG/2_7b_2bit

CUDA_VISIBLE_DEVICES=0 python3 \
    -m quip_sharp.eval.eval_ppl \
    --hf_path $HF/2_7b_2bit \
    2>&1 \
    | tee -a $LOG/2_7b_2bit

CUDA_VISIBLE_DEVICES=0 python3 \
    -m quip_sharp.eval.eval_zeroshot \
    --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    --batch_size 4 \
    2>&1 \
    --hf_path $HF/2_7b_2bit \
    | tee -a $LOG/2_7b_2bit

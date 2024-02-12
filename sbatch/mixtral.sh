#!/bin/bash

CKPT=/mnt/desa_data/checkpoints
HF=/mnt/desa_data/hfized/jerry
HESS=/mnt/desa_data/hessians
LOG=/mnt/desa_data/logs
L1=/mnt/desa_data/meta_llama1

source ~/miniconda3/bin/activate quipv2_mixtral

## mixtral 8192ctx
# python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx \
#     --codebook E8P12 --scale_override 0.9 --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --hessian_path $HESS/mixtral_8x7b_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx 2>&1
# python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx \
#     --codebook HI4B1C --scale_override 2.7 --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --hessian_path $HESS/mixtral_8x7b_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx 2>&1

## mixtral 12288ctx
# python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx \
#     --codebook E8P12 --scale_override 0.9 --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --hessian_path $HESS/mixtral_8x7b_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx 2>&1
# python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx \
#     --codebook HI4B1C --scale_override 2.7 --base_model mistralai/Mixtral-8x7B-v0.1 \
#     --hessian_path $HESS/mixtral_8x7b_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx 2>&1

## mixtral-instruct 8192ctx
# python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx \
#     --codebook E8P12 --scale_override 0.9 --base_model mistralai/Mixtral-8x7b-Instruct-v0.1 \
#     --hessian_path $HESS/mixtral_8x7b_instruct_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx 2>&1
# python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx \
#     --codebook HI4B1C --scale_override 2.7 --base_model mistralai/Mixtral-8x7b-Instruct-v0.1 \
#     --hessian_path $HESS/mixtral_8x7b_instruct_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx 2>&1
# 
## mixtral-instruct 12288ctx
python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx \
    --codebook E8P12 --scale_override 0.9 --base_model mistralai/Mixtral-8x7b-Instruct-v0.1 \
    --hessian_path $HESS/mixtral_8x7b_instruct_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx 2>&1
python quantize_mixtral.py --save_path $CKPT/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx \
    --codebook HI4B1C --scale_override 2.7 --base_model mistralai/Mixtral-8x7b-Instruct-v0.1 \
    --hessian_path $HESS/mixtral_8x7b_instruct_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx 2>&1

# CUDA_VISIBLE_DEVICES=0 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx \
#     --hf_output_path $HF/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx &
# wait

## hfize
## mixtral 8192ctx
# CUDA_VISIBLE_DEVICES=0 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx \
#     --hf_output_path $HF/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx & 
# CUDA_VISIBLE_DEVICES=0 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx \
#     --hf_output_path $HF/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx & 
# ## mixtral 12288ctx
# CUDA_VISIBLE_DEVICES=1 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx \
#     --hf_output_path $HF/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx & 
# CUDA_VISIBLE_DEVICES=2 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx \
#     --hf_output_path $HF/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx & 
# ## mixtral-instruct 8192ctx
# CUDA_VISIBLE_DEVICES=0 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx \
#     --hf_output_path $HF/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx & 
# CUDA_VISIBLE_DEVICES=1 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx \
#     --hf_output_path $HF/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx & 
## mixtral-instruct 12288ctx
CUDA_VISIBLE_DEVICES=0 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx \
    --hf_output_path $HF/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx & 
CUDA_VISIBLE_DEVICES=1 python hfize_mixtral.py --quantized_path $CKPT/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx \
    --hf_output_path $HF/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx & 
wait

## perplexity
## mixtral 8192ctx
# CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
#     --hf_path $HF/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
#     --hf_path $HF/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx 2>&1 &
# ## mixtral 12288ctx
# CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
#     --hf_path $HF/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
#     --hf_path $HF/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx 2>&1 &
# wait
# ## mixtral-instruct 8192ctx
# CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
#     --hf_path $HF/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
#     --hf_path $HF/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx 2>&1 &
## mixtral-instruct 12288ctx
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
    --hf_path $HF/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --no_use_cuda_graph --seqlen 8192 \
    --hf_path $HF/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx 2>&1 &
wait

## zeroshot
## mixtral 8192ctx
# CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     --batch_size 4 --hf_path $HF/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_e8p_2bit_RPv1_4096dev8192ctx 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     --batch_size 4 --hf_path $HF/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_hi_4bit_RPv1_4096dev8192ctx 2>&1 &
# ## mixtral 12288ctx
# CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     --batch_size 4 --hf_path $HF/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_e8p_2bit_RPv1_4096dev12288ctx 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     --batch_size 4 --hf_path $HF/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_hi_4bit_RPv1_4096dev12288ctx 2>&1 &
# ## mixtral-instruct 8192ctx
# CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     --batch_size 4 --hf_path $HF/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev8192ctx 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     --batch_size 4 --hf_path $HF/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx >> $LOG/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev8192ctx 2>&1 &
## mixtral-instruct 12288ctx
CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    --batch_size 4 --hf_path $HF/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_instruct_e8p_2bit_RPv1_4096dev12288ctx 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    --batch_size 4 --hf_path $HF/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx >> $LOG/mixtral_8x7b_instruct_hi_4bit_RPv1_4096dev12288ctx 2>&1 &
wait
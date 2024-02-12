#!/bin/bash

CKPT=/mnt/jerry_data/checkpoints
HF=/mnt/jerry_data/hfized
HESS=/mnt/jerry_data/hessians
LOG=/mnt/jerry_data/logs
L1=/mnt/jerry_data/meta_llama1

'''
CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python quantize_llama.py --save_path $CKPT/2_70b_hi_4bit  --codebook HI4B1C  --sigma_reg2 1e-2 --scale 2.7 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_hi_4bit 2>&1
CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 python quantize_llama.py --save_path $CKPT/2_13b_hi_4bit  --codebook HI4B1C  --sigma_reg2 1e-2 --scale 2.7 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 >> $LOG/2_13b_hi_4bit 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_hi_4bit   --codebook HI4B1C  --sigma_reg2 1e-2 --scale 2.7 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144  >> $LOG/2_7b_hi_4bit 2>&1
python quantize_llama.py --save_path $CKPT/1_65b_hi_4bit  --codebook HI4B1C  --sigma_reg2 1e-2 --scale 2.7 --base_model $L1/huggingface_65B --hessian_path $HESS/llama1_65b_6144 >> $LOG/1_65b_hi_4bit 2>&1
python quantize_llama.py --save_path $CKPT/1_30b_hi_4bit  --codebook HI4B1C  --sigma_reg2 1e-2 --scale 2.7 --base_model $L1/huggingface_30B --hessian_path $HESS/llama1_30b_6144 >> $LOG/1_30b_hi_4bit 2>&1
python quantize_llama.py --save_path $CKPT/1_13b_hi_4bit  --codebook HI4B1C  --sigma_reg2 1e-2 --scale 2.7 --base_model $L1/huggingface_13B --hessian_path $HESS/llama1_13b_6144 >> $LOG/1_13b_hi_4bit 2>&1
python quantize_llama.py --save_path $CKPT/1_7b_hi_4bit   --codebook HI4B1C  --sigma_reg2 1e-2 --scale 2.7 --base_model $L1/huggingface_7B  --hessian_path $HESS/llama1_7b_6144  >> $LOG/1_7b_hi_4bit 2>&1

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_hi_4bit  --hf_output_path $HF/2_70b_hi_4bit  &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_13b_hi_4bit  --hf_output_path $HF/2_13b_hi_4bit  &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_7b_hi_4bit   --hf_output_path $HF/2_7b_hi_4bit   &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_65b_hi_4bit  --hf_output_path $HF/1_65b_hi_4bit  &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/1_30b_hi_4bit  --hf_output_path $HF/1_30b_hi_4bit  &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/1_13b_hi_4bit  --hf_output_path $HF/1_13b_hi_4bit  &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/1_7b_hi_4bit   --hf_output_path $HF/1_7b_hi_4bit   &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --seqlen 2048 --hf_path $HF/2_70b_hi_4bit  >> $LOG/2_70b_hi_4bit  2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --seqlen 2048 --hf_path $HF/2_13b_hi_4bit  >> $LOG/2_13b_hi_4bit  2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --seqlen 2048 --hf_path $HF/2_7b_hi_4bit   >> $LOG/2_7b_hi_4bit   2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_65b_hi_4bit  >> $LOG/1_65b_hi_4bit  2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_30b_hi_4bit  >> $LOG/1_30b_hi_4bit  2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_13b_hi_4bit  >> $LOG/1_13b_hi_4bit  2>&1 &
CUDA_VISIBLE_DEVICES=6 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_7b_hi_4bit   >> $LOG/1_7b_hi_4bit   2>&1 &

wait
'''
CUDA_VISIBLE_DEVICES=0 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 1 --hf_path $HF/2_70b_hi_4bit  >> $LOG/2_70b_hi_4bit  2>&1 &
#CUDA_VISIBLE_DEVICES=1 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_hi_4bit  >> $LOG/2_13b_hi_4bit  2>&1 &
#CUDA_VISIBLE_DEVICES=2 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_hi_4bit   >> $LOG/2_7b_hi_4bit   2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 1 --hf_path $HF/1_65b_hi_4bit  >> $LOG/1_65b_hi_4bit  2>&1 &
#CUDA_VISIBLE_DEVICES=4 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_hi_4bit  >> $LOG/1_30b_hi_4bit  2>&1 &
#CUDA_VISIBLE_DEVICES=5 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_hi_4bit  >> $LOG/1_13b_hi_4bit  2>&1 &
#CUDA_VISIBLE_DEVICES=6 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_hi_4bit   >> $LOG/1_7b_hi_4bit   2>&1 &

wait
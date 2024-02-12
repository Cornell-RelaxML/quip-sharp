#!/bin/bash

CKPT=/mnt/jerry_data/checkpoints
HF=/mnt/jerry_data/hfized
HESS=/mnt/jerry_data/hessians
LOG=/mnt/jerry_data/logs

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python quantize_llama.py --save_path $CKPT/2_70b_hi_4bit_nolr --codebook HI4B1C --lora_rank 0 --scale_override 2.7 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_hi_4bit_nolr 2>&1

CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_70b_hi_4bit_nolr --hf_output_path $HF/2_70b_hi_4bit_nolr 

CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/2_70b_hi_4bit_nolr >> $LOG/2_70b_hi_4bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/2_13b_e8p_2bit_nolr >> $LOG/2_13b_e8p_2bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --hf_path $HF/2_7b_e8p_2bit_nolr  >> $LOG/2_7b_e8p_2bit_nolr  2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --hf_path $HF/2_13b_hi_4bit_nolr  >> $LOG/2_13b_hi_4bit_nolr  2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --hf_path $HF/2_7b_hi_4bit_nolr   >> $LOG/2_7b_hi_4bit_nolr   2>&1 &

wait

CUDA_VISIBLE_DEVICES=1 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_hi_4bit_nolr >> $LOG/2_70b_hi_4bit_nolr 2>&1


'''
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python quantize_llama.py --save_path $CKPT/2_13b_e8p_2bit_nolr --codebook E8P12 --lora_rank 0 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 >> $LOG/2_13b_e8p_2bit_nolr 2>&1
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python quantize_llama.py --save_path $CKPT/2_7b_e8p_2bit_nolr  --codebook E8P12 --lora_rank 0 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144  >> $LOG/2_7b_e8p_2bit_nolr 2>&1
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python quantize_llama.py --save_path $CKPT/2_13b_hi_4bit_nolr  --codebook HI4B1C  --lora_rank 0 --scale_override 2.7 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 >> $LOG/2_13b_hi_4bit_nolr 2>&1
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python quantize_llama.py --save_path $CKPT/2_7b_hi_4bit_nolr   --codebook HI4B1C  --lora_rank 0 --scale_override 2.7 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144  >> $LOG/2_7b_hi_4bit_nolr 2>&1


CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit_nolr --hf_output_path $HF/2_13b_e8p_2bit_nolr &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit_nolr  --hf_output_path $HF/2_7b_e8p_2bit_nolr  &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/2_13b_hi_4bit_nolr  --hf_output_path $HF/2_13b_hi_4bit_nolr  &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/2_7b_hi_4bit_nolr   --hf_output_path $HF/2_7b_hi_4bit_nolr   &

wait

CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --seqlen 2048 --hf_path $HF/2_13b_e8p_2bit_nolr >> $LOG/2_13b_e8p_2bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --seqlen 2048 --hf_path $HF/2_7b_e8p_2bit_nolr  >> $LOG/2_7b_e8p_2bit_nolr  2>&1 &
CUDA_VISIBLE_DEVICES=6 python ppl_llama.py --seqlen 2048 --hf_path $HF/2_13b_hi_4bit_nolr  >> $LOG/2_13b_hi_4bit_nolr  2>&1 &
CUDA_VISIBLE_DEVICES=7 python ppl_llama.py --seqlen 2048 --hf_path $HF/2_7b_hi_4bit_nolr   >> $LOG/2_7b_hi_4bit_nolr   2>&1 &

wait

CUDA_VISIBLE_DEVICES=2 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit_nolr >> $LOG/2_13b_e8p_2bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit_nolr  >> $LOG/2_7b_e8p_2bit_nolr  2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_hi_4bit_nolr  >> $LOG/2_13b_hi_4bit_nolr  2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_hi_4bit_nolr   >> $LOG/2_7b_hi_4bit_nolr   2>&1 &

wait

'''

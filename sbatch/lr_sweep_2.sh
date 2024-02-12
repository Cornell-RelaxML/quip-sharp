#!/bin/bash

CKPT=/mnt/jerry_data/checkpoints
HF=/mnt/jerry_data/hfized
HESS=/mnt/jerry_data/hessians
LOG=/mnt/jerry_data/logs


python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_fulllr4 --codebook E8P12 --scale_override 0.9 --lora_rank 4 --hessian_path $HESS/llama2_70b_6144 --full_svd >> $LOG/2_70b_e8p_2bit_fulllr4 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_fulllr2 --codebook E8P12 --scale_override 0.9 --lora_rank 2 --hessian_path $HESS/llama2_70b_6144 --full_svd >> $LOG/2_70b_e8p_2bit_fulllr2 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_fulllr1 --codebook E8P12 --scale_override 0.9 --lora_rank 1 --hessian_path $HESS/llama2_70b_6144 --full_svd >> $LOG/2_70b_e8p_2bit_fulllr1 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_nolr --codebook E8P12 --scale_override 0.9 --lora_rank 0 --hessian_path $HESS/llama2_70b_6144 --full_svd >> $LOG/2_70b_e8p_2bit_nolr 2>&1

wait

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_fulllr4 --hf_output_path $HF/2_70b_e8p_2bit_fulllr4 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_fulllr2 --hf_output_path $HF/2_70b_e8p_2bit_fulllr2 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_fulllr1 --hf_output_path $HF/2_70b_e8p_2bit_fulllr1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_nolr  --hf_output_path $HF/2_70b_e8p_2bit_nolr &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_fulllr4 >> $LOG/2_70b_e8p_2bit_fulllr4 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_fulllr2 >> $LOG/2_70b_e8p_2bit_fulllr2 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_fulllr1 >> $LOG/2_70b_e8p_2bit_fulllr1 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_nolr  >> $LOG/2_70b_e8p_2bit_nolr  2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_fulllr4 >> $LOG/2_70b_e8p_2bit_fulllr4 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_fulllr2 >> $LOG/2_70b_e8p_2bit_fulllr2 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_fulllr1 >> $LOG/2_70b_e8p_2bit_fulllr1 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_nolr  >> $LOG/2_70b_e8p_2bit_nolr  2>&1 &

wait

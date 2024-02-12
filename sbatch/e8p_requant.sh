#!/bin/bash

CKPT=/mnt/desa_data/checkpoints/new_e8p
HF=/mnt/desa_data/hfized/new_e8p
LOG=/mnt/desa_data/logs/new_e8p
HESS=/mnt/desa_data/hessians
L1=/mnt/desa_data/meta_llama1

mkdir $CKPT
mkdir $HF
mkdir $LOG

'''
# llama 2
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/2_13b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 >> $LOG/2_13b_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144 >> $LOG/2_7b_e8p_2bit 2>&1
# llama 2 chat
python quantize_llama.py --save_path $CKPT/2_70b_chat_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-chat-hf --hessian_path $HESS/llama2_70b_chat_6144 >> $LOG/2_70b_chat_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/2_13b_chat_e8p_2bit --codebook E8P12 --scale_override 1.0 --base_model meta-llama/Llama-2-13b-chat-hf --hessian_path $HESS/llama2_13b_chat_6144 >> $LOG/2_13b_chat_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_chat_e8p_2bit --codebook E8P12 --scale_override 1.0 --base_model meta-llama/Llama-2-7b-chat-hf --hessian_path $HESS/llama2_7b_chat_6144 >> $LOG/2_7b_chat_e8p_2bit 2>&1
# llama 1 
python quantize_llama.py --save_path $CKPT/1_65b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-65b-hf --hessian_path $HESS/llama1_65b_6144 >> $LOG/1_65b_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/1_30b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-30b-hf --hessian_path $HESS/llama1_30b_6144 >> $LOG/1_30b_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/1_13b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-13b-hf --hessian_path $HESS/llama1_13b_6144 >> $LOG/1_13b_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/1_7b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model relaxml/Llama-1-7b-hf --hessian_path $HESS/llama1_7b_6144 >> $LOG/1_7b_e8p_2bit 2>&1
# mistral hermes
python quantize_llama.py --save_path $CKPT/mistral_7b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model mistralai/Mistral-7B-v0.1 --hessian_path $HESS/mistral_7b_4096 >> $LOG/mistral_7b_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/openhermes_7b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model teknium/OpenHermes-2.5-Mistral-7B --hessian_path $HESS/openhermes_7b_4096 >> $LOG/openhermes_7b_e8p_2bit 2>&1


CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit --hf_output_path $HF/2_70b_e8p_2bit >> $LOG/2_70b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit --hf_output_path $HF/2_13b_e8p_2bit >> $LOG/2_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit --hf_output_path $HF/2_7b_e8p_2bit >> $LOG/2_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_chat_e8p_2bit --hf_output_path $HF/2_70b_chat_e8p_2bit >> $LOG/2_70b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_13b_chat_e8p_2bit --hf_output_path $HF/2_13b_chat_e8p_2bit >> $LOG/2_13b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/2_7b_chat_e8p_2bit --hf_output_path $HF/2_7b_chat_e8p_2bit >> $LOG/2_7b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/1_65b_e8p_2bit --hf_output_path $HF/1_65b_e8p_2bit >> $LOG/1_65b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/1_30b_e8p_2bit --hf_output_path $HF/1_30b_e8p_2bit >> $LOG/1_30b_e8p_2bit 2>&1 &

wait
CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/1_13b_e8p_2bit --hf_output_path $HF/1_13b_e8p_2bit >> $LOG/1_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/1_7b_e8p_2bit --hf_output_path $HF/1_7b_e8p_2bit >> $LOG/1_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/mistral_7b_e8p_2bit --hf_output_path $HF/mistral_7b_e8p_2bit >> $LOG/mistral_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/openhermes_7b_e8p_2bit --hf_output_path $HF/openhermes_7b_e8p_2bit >> $LOG/openhermes_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/2_70b_e8p_2bit >> $LOG/2_70b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/2_13b_e8p_2bit >> $LOG/2_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --hf_path $HF/2_7b_e8p_2bit >> $LOG/2_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --hf_path $HF/2_70b_chat_e8p_2bit >> $LOG/2_70b_chat_e8p_2bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/mistral_7b_e8p_2bit >> $LOG/mistral_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/openhermes_7b_e8p_2bit >> $LOG/openhermes_7b_e8p_2bit 2>&1 &

wait
'''

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_65b_e8p_2bit >> $LOG/1_65b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_30b_e8p_2bit >> $LOG/1_30b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_13b_e8p_2bit >> $LOG/1_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_7b_e8p_2bit >> $LOG/1_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_chat_e8p_2bit    >> $LOG/2_7b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_e8p_2bit    >> $LOG/1_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/mistral_7b_e8p_2bit    >> $LOG/mistral_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/openhermes_7b_e8p_2bit  >> $LOG/openhermes_7b_e8p_2bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit    >> $LOG/2_70b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit    >> $LOG/2_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit    >> $LOG/2_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_chat_e8p_2bit    >> $LOG/2_70b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_chat_e8p_2bit    >> $LOG/2_13b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_e8p_2bit    >> $LOG/1_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_e8p_2bit    >> $LOG/1_65b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_e8p_2bit    >> $LOG/1_30b_e8p_2bit 2>&1 &

wait

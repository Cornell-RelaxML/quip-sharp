#!/bin/bash

CKPT=/mnt/desa_data/checkpoints
HF=/mnt/desa_data/hfized
HESS=/mnt/desa_data/hessians
LOG=/mnt/desa_data/logs


python quantize_llama.py --save_path $CKPT/2_70b_chat_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-chat-hf --hessian_path $HESS/llama2_70b_chat_6144 >> $LOG/2_70b_chat_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/2_13b_chat_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-chat-hf --hessian_path $HESS/llama2_13b_chat_6144 >> $LOG/2_13b_chat_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_chat_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-chat-hf --hessian_path $HESS/llama2_7b_chat_6144 >> $LOG/2_7b_chat_e8p_2bit 2>&1

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_chat_e8p_2bit --hf_output_path $HF/2_70b_chat_e8p_2bit >> $LOG/2_70b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_13b_chat_e8p_2bit --hf_output_path $HF/2_13b_chat_e8p_2bit >> $LOG/2_13b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_7b_chat_e8p_2bit --hf_output_path $HF/2_7b_chat_e8p_2bit >> $LOG/2_7b_chat_e8p_2bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/2_70b_chat_e8p_2bit >> $LOG/2_70b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/2_13b_chat_e8p_2bit >> $LOG/2_13b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/2_7b_chat_e8p_2bit >> $LOG/2_7b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3,4 python ppl_llama.py --hf_path meta-llama/Llama-2-70b-chat-hf --no_use_cuda_graph >> $LOG/2_70b_chat_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --hf_path meta-llama/Llama-2-13b-chat-hf >> $LOG/2_13b_chat_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=6 python ppl_llama.py --hf_path meta-llama/Llama-2-7b-chat-hf >> $LOG/2_7b_chat_fp16 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_chat_e8p_2bit >> $LOG/2_70b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_chat_e8p_2bit >> $LOG/2_13b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_chat_e8p_2bit >> $LOG/2_7b_chat_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3,4 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 2  --hf_path meta-llama/Llama-2-70b-chat-hf >> $LOG/2_70b_chat_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path meta-llama/Llama-2-13b-chat-hf >> $LOG/2_13b_chat_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path meta-llama/Llama-2-7b-chat-hf >> $LOG/2_7b_chat_fp16 2>&1 &
wait					     

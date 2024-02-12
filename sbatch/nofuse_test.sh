#!/bin/bash

CKPT=/mnt/desa_data/checkpoints
HF=/mnt/desa_data/hfized
HESS=/mnt/desa_data/hessians
LOG=/mnt/desa_data/logs
L1=/mnt/desa_data/meta_llama1

'''
python quantize_llama_nofuse.py --save_path $CKPT/2_70b_e8p_2bit_nofuse --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_e8p_2bit_nofuse 2>&1
python quantize_llama_nofuse.py --save_path $CKPT/2_13b_e8p_2bit_nofuse --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 >> $LOG/2_13b_e8p_2bit_nofuse 2>&1
python quantize_llama_nofuse.py --save_path $CKPT/2_7b_e8p_2bit_nofuse  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144  >> $LOG/2_7b_e8p_2bit_nofuse 2>&1
python quantize_llama_nofuse.py --save_path $CKPT/1_65b_e8p_2bit_nofuse --codebook E8P12 --scale_override 0.9 --base_model $L1/huggingface_65B --hessian_path $HESS/llama1_65b_6144 >> $LOG/1_65b_e8p_2bit_nofuse 2>&1
python quantize_llama_nofuse.py --save_path $CKPT/1_30b_e8p_2bit_nofuse --codebook E8P12 --scale_override 0.9 --base_model $L1/huggingface_30B --hessian_path $HESS/llama1_30b_6144 >> $LOG/1_30b_e8p_2bit_nofuse 2>&1
python quantize_llama_nofuse.py --save_path $CKPT/1_13b_e8p_2bit_nofuse --codebook E8P12 --scale_override 0.9 --base_model $L1/huggingface_13B --hessian_path $HESS/llama1_13b_6144 >> $LOG/1_13b_e8p_2bit_nofuse 2>&1
python quantize_llama_nofuse.py --save_path $CKPT/1_7b_e8p_2bit_nofuse  --codebook E8P12 --scale_override 0.9 --base_model $L1/huggingface_7B  --hessian_path $HESS/llama1_7b_6144  >> $LOG/1_7b_e8p_2bit_nofuse 2>&1


CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_nofuse --hf_output_path $HF/2_70b_e8p_2bit_nofuse >> $LOG/2_70b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit_nofuse --hf_output_path $HF/2_13b_e8p_2bit_nofuse >> $LOG/2_13b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit_nofuse --hf_output_path $HF/2_7b_e8p_2bit_nofuse >> $LOG/2_7b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_65b_e8p_2bit_nofuse --hf_output_path $HF/1_65b_e8p_2bit_nofuse >> $LOG/1_65b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/1_30b_e8p_2bit_nofuse --hf_output_path $HF/1_30b_e8p_2bit_nofuse >> $LOG/1_30b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/1_13b_e8p_2bit_nofuse --hf_output_path $HF/1_13b_e8p_2bit_nofuse >> $LOG/1_13b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/1_7b_e8p_2bit_nofuse --hf_output_path $HF/1_7b_e8p_2bit_nofuse >> $LOG/1_7b_e8p_2bit_nofuse 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_nofuse >> $LOG/2_70b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/2_13b_e8p_2bit_nofuse >> $LOG/2_13b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/2_7b_e8p_2bit_nofuse >> $LOG/2_7b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_65b_e8p_2bit_nofuse >> $LOG/1_65b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_30b_e8p_2bit_nofuse >> $LOG/1_30b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_13b_e8p_2bit_nofuse >> $LOG/1_13b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=6 python ppl_llama.py --seqlen 2048 --hf_path $HF/1_7b_e8p_2bit_nofuse >> $LOG/1_7b_e8p_2bit_nofuse 2>&1 &

wait
'''
CUDA_VISIBLE_DEVICES=0 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_nofuse >> $LOG/2_70b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit_nofuse >> $LOG/2_13b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit_nofuse >> $LOG/2_7b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_e8p_2bit_nofuse >> $LOG/1_65b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_e8p_2bit_nofuse >> $LOG/1_30b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_e8p_2bit_nofuse >> $LOG/1_13b_e8p_2bit_nofuse 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_e8p_2bit_nofuse >> $LOG/1_7b_e8p_2bit_nofuse 2>&1 &

wait

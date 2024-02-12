#!/bin/bash

CKPT=/mnt/desa_data/checkpoints
HF=/mnt/desa_data/hfized
HESS=/mnt/desa_data/hessians
LOG=/mnt/desa_data/logs
L1=/mnt/desa_data/meta_llama1

'''
python quantize_llama.py --save_path $CKPT/mistral_7b_hi_4bit_packed --codebook HI4B1C --scale_override 2.7 --base_model mistralai/Mistral-7B-v0.1 --hessian_path $HESS/mistral_7b_4096 >> $LOG/mistral_7b_hi_4bit_packed 2>&1
python quantize_llama.py --save_path $CKPT/openhermes_7b_hi_4bit_packed --codebook HI4B1C --scale_override 2.7 --base_model teknium/OpenHermes-2.5-Mistral-7B --hessian_path $HESS/openhermes_7b_4096 >> $LOG/openhermes_7b_hi_4bit_packed 2>&1
python quantize_llama.py --save_path $CKPT/mistral_7b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model mistralai/Mistral-7B-v0.1 --hessian_path $HESS/mistral_7b_4096 >> $LOG/mistral_7b_e8p_2bit 2>&1
python quantize_llama.py --save_path $CKPT/openhermes_7b_e8p_2bit --codebook E8P12 --scale_override 0.9 --base_model teknium/OpenHermes-2.5-Mistral-7B --hessian_path $HESS/openhermes_7b_4096 >> $LOG/openhermes_7b_e8p_2bit 2>&1

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/mistral_7b_hi_4bit_packed    --hf_output_path $HF/mistral_7b_hi_4bit_packed    & 
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/openhermes_7b_hi_4bit_packed --hf_output_path $HF/openhermes_7b_hi_4bit_packed &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/mistral_7b_e8p_2bit          --hf_output_path $HF/mistral_7b_e8p_2bit          &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/openhermes_7b_e8p_2bit       --hf_output_path $HF/openhermes_7b_e8p_2bit       &

wait
'''
# perplexity
CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --no_use_cuda_graph --seqlen 8192 --hf_path $HF/mistral_7b_hi_4bit_packed    >> $LOG/mistral_7b_hi_4bit_packed    2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --no_use_cuda_graph --seqlen 8192 --hf_path $HF/openhermes_7b_hi_4bit_packed >> $LOG/openhermes_7b_hi_4bit_packed 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --no_use_cuda_graph --seqlen 8192 --hf_path $HF/mistral_7b_e8p_2bit          >> $LOG/mistral_7b_e8p_2bit          2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --no_use_cuda_graph --seqlen 8192 --hf_path $HF/openhermes_7b_e8p_2bit       >> $LOG/openhermes_7b_e8p_2bit       2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --no_use_cuda_graph --seqlen 8192 --hf_path mistralai/Mistral-7B-v0.1        >> $LOG/mistral_7b_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --no_use_cuda_graph --seqlen 8192 --hf_path teknium/OpenHermes-2.5-Mistral-7B >> $LOG/openhermes_7b_fp16 2>&1 &

wait
'''
CUDA_VISIBLE_DEVICES=0 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/mistral_7b_hi_4bit_packed    >> $LOG/mistral_7b_hi_4bit_packed    2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/openhermes_7b_hi_4bit_packed >> $LOG/openhermes_7b_hi_4bit_packed 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/mistral_7b_e8p_2bit          >> $LOG/mistral_7b_e8p_2bit          2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/openhermes_7b_e8p_2bit       >> $LOG/openhermes_7b_e8p_2bit       2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path mistralai/Mistral-7B-v0.1        >> $LOG/mistral_7b_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path teknium/OpenHermes-2.5-Mistral-7B >> $LOG/openhermes_7b_fp16 2>&1 &

wait
'''

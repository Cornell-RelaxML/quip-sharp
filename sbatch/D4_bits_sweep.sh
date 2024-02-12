#!/bin/bash

CKPT=/mnt/jerry_data/checkpoints
HF=/mnt/jerry_data/hfized
HESS=/mnt/jerry_data/hessians
LOG=/mnt/jerry_data/logs
L1=/mnt/jerry_data/meta_llama1


python quantize_llama.py --save_path $CKPT/2_70b_d4_2bit_nolr --codebook D4 --lora_rank 0 --scale_override 1.1 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_d4_2bit_nolr 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_d4_221bit_nolr --codebook D4221B --lora_rank 0 --scale_override 1.2 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_d4_221bit_nolr 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_d4_234bit_nolr --codebook D4234B --lora_rank 0 --scale_override 1.4 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_d4_234bit_nolr 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_d4_248bit_nolr --codebook D4248B --lora_rank 0 --scale_override 1.4 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_d4_248bit_nolr 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_d4_274bit_nolr --codebook D4274B --lora_rank 0 --scale_override 1.6 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_d4_274bit_nolr 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_d4_299bit_nolr --codebook D4299B --lora_rank 0 --scale_override 1.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 >> $LOG/2_70b_d4_299bit_nolr 2>&1

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_d4_2bit_nolr --hf_output_path $HF/2_70b_d4_2bit_nolr & 
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_70b_d4_221bit_nolr --hf_output_path $HF/2_70b_d4_221bit_nolr & 
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_70b_d4_234bit_nolr --hf_output_path $HF/2_70b_d4_234bit_nolr & 
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_d4_248bit_nolr --hf_output_path $HF/2_70b_d4_248bit_nolr & 
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_70b_d4_274bit_nolr --hf_output_path $HF/2_70b_d4_274bit_nolr & 
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/2_70b_d4_299bit_nolr --hf_output_path $HF/2_70b_d4_299bit_nolr & 

wait

# perplexity
CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --seqlen 4096 --hf_path $HF/2_70b_d4_2bit_nolr >> $LOG/2_70b_d4_2bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --seqlen 4096 --hf_path $HF/2_70b_d4_221bit_nolr >> $LOG/2_70b_d4_221bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --seqlen 4096 --hf_path $HF/2_70b_d4_234bit_nolr >> $LOG/2_70b_d4_234bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --seqlen 4096 --hf_path $HF/2_70b_d4_248bit_nolr >> $LOG/2_70b_d4_248bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --seqlen 4096 --hf_path $HF/2_70b_d4_274bit_nolr >> $LOG/2_70b_d4_274bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --seqlen 4096 --hf_path $HF/2_70b_d4_299bit_nolr >> $LOG/2_70b_d4_299bit_nolr 2>&1 &

wait

# zero shot
CUDA_VISIBLE_DEVICES=0 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_d4_2bit_nolr >> $LOG/2_70b_d4_2bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_d4_221bit_nolr >> $LOG/2_70b_d4_221bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_d4_234bit_nolr >> $LOG/2_70b_d4_234bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_d4_248bit_nolr >> $LOG/2_70b_d4_248bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_d4_274bit_nolr >> $LOG/2_70b_d4_274bit_nolr 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_d4_299bit_nolr >> $LOG/2_70b_d4_299bit_nolr 2>&1 &

wait

#!/bin/bash

CKPT=/mnt/jerry_data/checkpoints
HF=/mnt/jerry_data/hfized
HESS=/mnt/jerry_data/hessians
LOG=/mnt/jerry_data/logs

NAME=2_70b_e8_237bit_nolr

CUDA_VISIBLE_DEVICES=4,5,6,7 python quantize_llama.py --save_path $CKPT/$NAME --codebook E8237B --scale_override 1.13 --lora_rank 0 --hessian_path $HESS/llama2_70b_6144 >> $LOG/$NAME 2>&1

CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/$NAME --hf_output_path $HF/$NAME

CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --hf_path $HF/$NAME >> $LOG/$NAME 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/$NAME >> $LOG/$NAME 2>&1 &

wait


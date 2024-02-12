#!/bin/bash

CKPT=/mnt/jerry_data/checkpoints
HF=/mnt/jerry_data/hfized
HESS=/mnt/jerry_data/hessians
LOG=/mnt/jerry_data/logs
NAME=2_70b_e8p_2bit

python quantize_llama.py --save_path $CKPT/$NAME --codebook E8P12 --sigma_reg2 1e-2 --scale 0.90 --hessian_path $HESS/llama2_70b_6144 >> $LOG/$NAME 2>&1
CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/$NAME --hf_output_path $HF/$NAME >> $LOG/$NAME 2>&1
CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/$NAME >> $LOG/$NAME 2>&1
CUDA_VISIBLE_DEVICES=0 python eval_llama.py --hf_path $HF/$NAME --batch_size 4 --tasks arc_challenge,arc_easy,boolq,piqa,winogrande >> $LOG/$NAME 2>&1

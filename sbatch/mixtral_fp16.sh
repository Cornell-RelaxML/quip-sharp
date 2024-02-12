#!/bin/bash

LOG=/mnt/desa_data/logs

source ~/miniconda3/bin/activate quipv2_mixtral

# python eval_ppl.py --no_use_cuda_graph --seqlen 8192 --hf_path mistralai/Mixtral-8x7B-v0.1 >> $LOG/mixtral_8x7b_fp16 2>&1
python eval_ppl.py --no_use_cuda_graph --seqlen 8192 --hf_path mistralai/Mixtral-8x7B-Instruct-v0.1 >> $LOG/mixtral_8x7b_instruct_fp16 2>&1

# python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
#     --batch_size 2 --hf_path mistralai/Mixtral-8x7B-v0.1 >> $LOG/mixtral_8x7b_fp16 2>&1
python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande \
    --batch_size 2 --hf_path mistralai/Mixtral-8x7B-Instruct-v0.1 >> $LOG/mixtral_8x7b_instruct_fp16 2>&1
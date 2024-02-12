#!/bin/bash

CKPT=/mnt/desa_data/checkpoints
HF=/mnt/desa_data/hfized
HESS=/mnt/desa_data/hessians
LOG=/mnt/desa_data/logs

'''
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_gr0  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 --quip_tune_iters 0 >> $LOG/2_70b_e8p_2bit_gr0 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_gr5  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 --quip_tune_iters 5 >> $LOG/2_70b_e8p_2bit_gr5 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_gr10 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 --quip_tune_iters 10 >> $LOG/2_70b_e8p_2bit_gr10 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_gr15 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 --quip_tune_iters 15 >> $LOG/2_70b_e8p_2bit_gr15 2>&1
python quantize_llama.py --save_path $CKPT/2_70b_e8p_2bit_gr20 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf --hessian_path $HESS/llama2_70b_6144 --quip_tune_iters 20 >> $LOG/2_70b_e8p_2bit_gr20 2>&1


CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_gr0   --hf_output_path $HF/2_70b_e8p_2bit_gr0  >> $LOG/2_70b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_gr5   --hf_output_path $HF/2_70b_e8p_2bit_gr5  >> $LOG/2_70b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_gr10  --hf_output_path $HF/2_70b_e8p_2bit_gr10 >> $LOG/2_70b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_gr15  --hf_output_path $HF/2_70b_e8p_2bit_gr15 >> $LOG/2_70b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_70b_e8p_2bit_gr20  --hf_output_path $HF/2_70b_e8p_2bit_gr20 >> $LOG/2_70b_e8p_2bit_gr20 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_gr0  >> $LOG/2_70b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_gr5  >> $LOG/2_70b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_gr10 >> $LOG/2_70b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_gr15 >> $LOG/2_70b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --hf_path $HF/2_70b_e8p_2bit_gr20 >> $LOG/2_70b_e8p_2bit_gr20 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_gr0  >> $LOG/2_70b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_gr5  >> $LOG/2_70b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_gr10 >> $LOG/2_70b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_gr15 >> $LOG/2_70b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_e8p_2bit_gr20 >> $LOG/2_70b_e8p_2bit_gr20 2>&1 &
wait					     
'''

python quantize_llama.py --save_path $CKPT/2_13b_e8p_2bit_gr0  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 --quip_tune_iters 0 >> $LOG/2_13b_e8p_2bit_gr0 2>&1
python quantize_llama.py --save_path $CKPT/2_13b_e8p_2bit_gr5  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 --quip_tune_iters 5 >> $LOG/2_13b_e8p_2bit_gr5 2>&1
python quantize_llama.py --save_path $CKPT/2_13b_e8p_2bit_gr10 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 --quip_tune_iters 10 >> $LOG/2_13b_e8p_2bit_gr10 2>&1
python quantize_llama.py --save_path $CKPT/2_13b_e8p_2bit_gr15 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 --quip_tune_iters 15 >> $LOG/2_13b_e8p_2bit_gr15 2>&1
python quantize_llama.py --save_path $CKPT/2_13b_e8p_2bit_gr20 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf --hessian_path $HESS/llama2_13b_6144 --quip_tune_iters 20 >> $LOG/2_13b_e8p_2bit_gr20 2>&1


CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit_gr0   --hf_output_path $HF/2_13b_e8p_2bit_gr0  >> $LOG/2_13b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit_gr5   --hf_output_path $HF/2_13b_e8p_2bit_gr5  >> $LOG/2_13b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit_gr10  --hf_output_path $HF/2_13b_e8p_2bit_gr10 >> $LOG/2_13b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit_gr15  --hf_output_path $HF/2_13b_e8p_2bit_gr15 >> $LOG/2_13b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_13b_e8p_2bit_gr20  --hf_output_path $HF/2_13b_e8p_2bit_gr20 >> $LOG/2_13b_e8p_2bit_gr20 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/2_13b_e8p_2bit_gr0  >> $LOG/2_13b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/2_13b_e8p_2bit_gr5  >> $LOG/2_13b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/2_13b_e8p_2bit_gr10 >> $LOG/2_13b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --hf_path $HF/2_13b_e8p_2bit_gr15 >> $LOG/2_13b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --hf_path $HF/2_13b_e8p_2bit_gr20 >> $LOG/2_13b_e8p_2bit_gr20 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit_gr0  >> $LOG/2_13b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit_gr5  >> $LOG/2_13b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit_gr10 >> $LOG/2_13b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit_gr15 >> $LOG/2_13b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_e8p_2bit_gr20 >> $LOG/2_13b_e8p_2bit_gr20 2>&1 &
wait


python quantize_llama.py --save_path $CKPT/2_7b_e8p_2bit_gr0  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144 --quip_tune_iters 0 >> $LOG/2_7b_e8p_2bit_gr0 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_e8p_2bit_gr5  --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144 --quip_tune_iters 5 >> $LOG/2_7b_e8p_2bit_gr5 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_e8p_2bit_gr10 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144 --quip_tune_iters 10 >> $LOG/2_7b_e8p_2bit_gr10 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_e8p_2bit_gr15 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144 --quip_tune_iters 15 >> $LOG/2_7b_e8p_2bit_gr15 2>&1
python quantize_llama.py --save_path $CKPT/2_7b_e8p_2bit_gr20 --codebook E8P12 --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf --hessian_path $HESS/llama2_7b_6144 --quip_tune_iters 20 >> $LOG/2_7b_e8p_2bit_gr20 2>&1


CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit_gr0   --hf_output_path $HF/2_7b_e8p_2bit_gr0  >> $LOG/2_7b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit_gr5   --hf_output_path $HF/2_7b_e8p_2bit_gr5  >> $LOG/2_7b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit_gr10  --hf_output_path $HF/2_7b_e8p_2bit_gr10 >> $LOG/2_7b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit_gr15  --hf_output_path $HF/2_7b_e8p_2bit_gr15 >> $LOG/2_7b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_7b_e8p_2bit_gr20  --hf_output_path $HF/2_7b_e8p_2bit_gr20 >> $LOG/2_7b_e8p_2bit_gr20 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/2_7b_e8p_2bit_gr0  >> $LOG/2_7b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/2_7b_e8p_2bit_gr5  >> $LOG/2_7b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/2_7b_e8p_2bit_gr10 >> $LOG/2_7b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --hf_path $HF/2_7b_e8p_2bit_gr15 >> $LOG/2_7b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --hf_path $HF/2_7b_e8p_2bit_gr20 >> $LOG/2_7b_e8p_2bit_gr20 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit_gr0  >> $LOG/2_7b_e8p_2bit_gr0  2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit_gr5  >> $LOG/2_7b_e8p_2bit_gr5  2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit_gr10 >> $LOG/2_7b_e8p_2bit_gr10 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit_gr15 >> $LOG/2_7b_e8p_2bit_gr15 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_llama.py  --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_e8p_2bit_gr20 >> $LOG/2_7b_e8p_2bit_gr20 2>&1 &
wait

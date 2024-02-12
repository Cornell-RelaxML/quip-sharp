#!/bin/bash

CKPT=checkpoints
HF=hfized
'''
CUDA_VISIBLE_DEVICES=0 python quantize_llama.py --save_path $CKPT/d4_s110  --codebook D4 --sigma_reg2 1e-2 --scale 1.10 --hessian_path hessians/llama2_70b_6144 > slurm_out/d4_s110.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python quantize_llama.py --save_path $CKPT/d4_s115  --codebook D4 --sigma_reg2 1e-2 --scale 1.15 --hessian_path hessians/llama2_70b_6144 > slurm_out/d4_s115.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python quantize_llama.py --save_path $CKPT/d4_s121  --codebook D4 --sigma_reg2 1e-2 --scale 1.21 --hessian_path hessians/llama2_70b_6144 > slurm_out/d4_s121.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python quantize_llama.py --save_path $CKPT/d4_s120  --codebook D4 --sigma_reg2 1e-2 --scale 1.20 --hessian_path hessians/llama2_70b_6144 > slurm_out/d4_s120.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python quantize_llama.py --save_path $CKPT/d4_s125  --codebook D4 --sigma_reg2 1e-2 --scale 1.25 --hessian_path hessians/llama2_70b_6144 > slurm_out/d4_s125.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python quantize_llama.py --save_path $CKPT/d4_s130  --codebook D4 --sigma_reg2 1e-2 --scale 1.30 --hessian_path hessians/llama2_70b_6144 > slurm_out/d4_s130.log 2>&1 &

wait
'''
CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/d4_s110  --hf_output_path $HF/d4_s110  >> slurm_out/d4_s110.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/d4_s115  --hf_output_path $HF/d4_s115  >> slurm_out/d4_s115.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/d4_s121  --hf_output_path $HF/d4_s121  >> slurm_out/d4_s121.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/d4_s120  --hf_output_path $HF/d4_s120  >> slurm_out/d4_s120.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/d4_s125  --hf_output_path $HF/d4_s125  >> slurm_out/d4_s125.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/d4_s130 --hf_output_path $HF/d4_s130 >> slurm_out/d4_s130.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/d4_s110   >> slurm_out/d4_s110.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/d4_s115   >> slurm_out/d4_s115.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/d4_s121   >> slurm_out/d4_s121.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --hf_path $HF/d4_s120   >> slurm_out/d4_s120.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --hf_path $HF/d4_s125   >> slurm_out/d4_s125.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --hf_path $HF/d4_s130   >> slurm_out/d4_s130.log 2>&1 &

wait

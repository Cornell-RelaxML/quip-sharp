#!/bin/bash

CKPT=checkpoints
HF=hfized
'''
CUDA_VISIBLE_DEVICES=0 python quantize_llama.py --save_path $CKPT/e8p_s075  --codebook E8P12 --sigma_reg2 1e-2 --scale 0.75 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s075.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python quantize_llama.py --save_path $CKPT/e8p_s080  --codebook E8P12 --sigma_reg2 1e-2 --scale 0.80 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s080.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python quantize_llama.py --save_path $CKPT/e8p_s085  --codebook E8P12 --sigma_reg2 1e-2 --scale 0.85 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s085.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python quantize_llama.py --save_path $CKPT/e8p_s090  --codebook E8P12 --sigma_reg2 1e-2 --scale 0.90 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s090.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python quantize_llama.py --save_path $CKPT/e8p_s095  --codebook E8P12 --sigma_reg2 1e-2 --scale 0.95 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s095.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python quantize_llama.py --save_path $CKPT/e8p_s0100 --codebook E8P12 --sigma_reg2 1e-2 --scale 1.00 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s100.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python quantize_llama.py --save_path $CKPT/e8p_s0105 --codebook E8P12 --sigma_reg2 1e-2 --scale 1.05 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s105.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python quantize_llama.py --save_path $CKPT/e8p_s0103 --codebook E8P12 --sigma_reg2 1e-2 --scale 1.03 --hessian_path hessians/llama2_70b_6144 > slurm_out/e8p_s103.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/e8p_s075  --hf_output_path $HF/e8p_s075  >> slurm_out/e8p_s075.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/e8p_s080  --hf_output_path $HF/e8p_s080  >> slurm_out/e8p_s080.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/e8p_s085  --hf_output_path $HF/e8p_s085  >> slurm_out/e8p_s085.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/e8p_s090  --hf_output_path $HF/e8p_s090  >> slurm_out/e8p_s090.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/e8p_s095  --hf_output_path $HF/e8p_s095  >> slurm_out/e8p_s095.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/e8p_s0100 --hf_output_path $HF/e8p_s0100 >> slurm_out/e8p_s100.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/e8p_s0105 --hf_output_path $HF/e8p_s0105 >> slurm_out/e8p_s105.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/e8p_s0103 --hf_output_path $HF/e8p_s0103 >> slurm_out/e8p_s103.log 2>&1 &

wait
'''
CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path $HF/e8p_s075 >> slurm_out/e8p_s075.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ppl_llama.py --hf_path $HF/e8p_s080 >> slurm_out/e8p_s080.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ppl_llama.py --hf_path $HF/e8p_s085 >> slurm_out/e8p_s085.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python ppl_llama.py --hf_path $HF/e8p_s090 >> slurm_out/e8p_s090.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ppl_llama.py --hf_path $HF/e8p_s095 >> slurm_out/e8p_s095.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python ppl_llama.py --hf_path $HF/e8p_s0100 >> slurm_out/e8p_s100.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python ppl_llama.py --hf_path $HF/e8p_s0105 >> slurm_out/e8p_s105.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python ppl_llama.py --hf_path $HF/e8p_s0103 >> slurm_out/e8p_s103.log 2>&1 &

wait

#!/bin/bash

CKPT=/mnt/desa_data/checkpoints
HF=/mnt/desa_data/hfized
HESS=/mnt/desa_data/hessians
LOG=/mnt/desa_data/logs
L1=/mnt/desa_data/meta_llama1

function sc_sweep {
    # NPRE $1
    # BMO $2
    SC_LS=("0.80" "0.85" "0.90" "0.95" "1.00")
    NPOST_LS=("080" "085" "090" "095" "100")
    for idx in "${!SC_LS[@]}"
    do
    python quantize_llama.py --save_path $CKPT/${1}_e8p_2bit_sc${NPOST_LS[$idx]} --codebook E8P12 --scale_override ${SC_LS[$idx]} \
        --base_model meta-llama/$2 --hessian_path $HESS/llama${1}_6144 >> $LOG/${1}_e8p_2bit_sc${NPOST_LS[$idx]} 2>&1
    done
    for idx in "${!SC_LS[@]}"
    do
    CUDA_VISIBLE_DEVICES=$idx python hfize_llama.py --quantized_path $CKPT/${1}_e8p_2bit_sc${NPOST_LS[$idx]} \
        --hf_output_path $HF/${1}_e8p_2bit_sc${NPOST_LS[$idx]} &
    done
    wait
    # perplexity
    for idx in "${!SC_LS[@]}"
    do
    CUDA_VISIBLE_DEVICES=$idx python ppl_llama.py --seqlen 4096 --hf_path $HF/${1}_e8p_2bit_sc${NPOST_LS[$idx]} \
        >> $LOG/${1}_e8p_2bit_sc${NPOST_LS[$idx]} 2>&1 &
    done
    wait 
    # zeroshot
    for idx in "${!SC_LS[@]}"
    do
    CUDA_VISIBLE_DEVICES=$idx python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 \
        --hf_path $HF/${1}_e8p_2bit_sc${NPOST_LS[$idx]} >> $LOG/${1}_e8p_2bit_sc${NPOST_LS[$idx]} 2>&1 &
    done
}

# sc_sweep "2_70b" "Llama-2-70b-hf"
# sc_sweep "2_13b" "Llama-2-13b-hf"
# sc_sweep "2_7b"  "Llama-2-7b-hf"
# 
# sc_sweep "2_70b_chat" "Llama-2-70b-chat-hf"
# sc_sweep "2_13b_chat" "Llama-2-13b-chat-hf"
# sc_sweep "2_7b_chat"  "Llama-2-7b-chat-hf"

function sc_sweep_hi {
    # NPRE $1
    # BMO $2
    SC_LS=("2.4" "2.55" "2.7" "2.85" "3")
    NPOST_LS=("240" "255" "270" "285" "300")
    for idx in "${!SC_LS[@]}"
    do
    python quantize_llama.py --save_path $CKPT/${1}_hi_4bit_sc${NPOST_LS[$idx]} --codebook HI4B1C --scale_override ${SC_LS[$idx]} \
        --base_model meta-llama/$2 --hessian_path $HESS/llama${1}_6144 >> $LOG/${1}_hi_4bit_sc${NPOST_LS[$idx]} 2>&1
    done
    for idx in "${!SC_LS[@]}"
    do
    CUDA_VISIBLE_DEVICES=$idx python hfize_llama.py --quantized_path $CKPT/${1}_hi_4bit_sc${NPOST_LS[$idx]} \
        --hf_output_path $HF/${1}_hi_4bit_sc${NPOST_LS[$idx]} &
    done
    wait
    # perplexity
    for idx in "${!SC_LS[@]}"
    do
    CUDA_VISIBLE_DEVICES=$idx python ppl_llama.py --seqlen 4096 --hf_path $HF/${1}_hi_4bit_sc${NPOST_LS[$idx]} \
        >> $LOG/${1}_hi_4bit_sc${NPOST_LS[$idx]} 2>&1 &
    done
    wait 
    # zeroshot
    for idx in "${!SC_LS[@]}"
    do
    CUDA_VISIBLE_DEVICES=$idx python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 \
        --hf_path $HF/${1}_hi_4bit_sc${NPOST_LS[$idx]} >> $LOG/${1}_hi_4bit_sc${NPOST_LS[$idx]} 2>&1 &
    done
}

sc_sweep_hi "2_70b" "Llama-2-70b-hf"
sc_sweep_hi "2_13b" "Llama-2-13b-hf"
sc_sweep_hi "2_7b"  "Llama-2-7b-hf"

sc_sweep_hi "2_70b_chat" "Llama-2-70b-chat-hf"
sc_sweep_hi "2_13b_chat" "Llama-2-13b-chat-hf"
sc_sweep_hi "2_7b_chat"  "Llama-2-7b-chat-hf"
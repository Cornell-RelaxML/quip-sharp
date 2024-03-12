# Example script for quantizing Llama 2 7b with QuIP#

CKPT=ckpt
HF=hf
LOG=log
HESS=hess

mkdir $CKPT
mkdir $HF
mkdir $LOG

# generate hessians (takes a while, only use this if there aren't pregenerated hessians)
# python hessian_offline.py --batch_size 4 --devset_size 6144 --ctx_size 2048 --base_model meta-llama/Llama-2-7b-hf --save_path $HESS/llama2_7b_6144

# download hessians (see the relaxml HF org to see what we have already)
# python scripts/download_hf.py --folder_path $HESS/llama2_7b_6144 --repo_id relaxml/Hessians-Llama-2-7b-6144 --read_token <TOKEN>

# quantize with finetuning
python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/2_7b_2bit --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144/ --devset_size 384 --ft_valid_size 128 >> $LOG/2_7b_2bit 2>&1

# convert model to hf format for end to end fine tuning
CUDA_VISIBLE_DEVICES=0 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_7b_2bit --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1

# end to end fine tuning
python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_2bit --devset_size 384 --ft_valid_size 128 --ft_epochs 8  --ft_bs 1 --ctx_size 4096 --ft_update_freq 2 --ft_train_mode --ckpt_path $CKPT/2_7b_2bit >> $LOG/2_7b_2bit 2>&1

# eval
CUDA_VISIBLE_DEVICES=0 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_7b_2bit --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_ppl --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 &  





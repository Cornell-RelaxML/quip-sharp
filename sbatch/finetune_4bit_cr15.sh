CKPT=/mnt/desa_data/checkpoints/finetune_llama
HF=/mnt/desa_data/hfized/finetune_llama
LOG=/mnt/desa_data/logs/finetune_llama
HESS=/mnt/desa_data/hessians



CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model meta-llama/Llama-2-13b-hf --hf_path $HF/2_13b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 4096 --ft_update_freq 2 --ckpt_path $CKPT/2_13b_4bit_scale >> $LOG/2_13b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 python tune_susv_lmhead.py --base_model relaxml/Llama-1-7b-hf --hf_path $HF/1_7b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_7b_4bit_scale >> $LOG/1_7b_4bit_scale 2>&1 &
wait

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-30b-hf --hf_path $HF/1_30b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_30b_4bit_scale >> $LOG/1_30b_4bit_scale 2>&1


CUDA_VISIBLE_DEVICES=2,3,4 python tune_susv_lmhead.py --base_model relaxml/Llama-1-13b-hf --hf_path $HF/1_13b_3bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_13b_3bit_scale >> $LOG/1_13b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-13b-hf --hf_path $HF/1_13b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_13b_4bit_scale >> $LOG/1_13b_4bit_scale 2>&1 &
wait

CUDA_VISIBLE_DEVICES=5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-7b-hf --hf_path $HF/1_7b_3bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_7b_3bit_scale >> $LOG/1_7b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=2,3,4 python tune_susv_lmhead.py --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 4096 --ft_update_freq 2 --ckpt_path $CKPT/2_7b_4bit_scale >> $LOG/2_7b_4bit_scale 2>&1 &
wait


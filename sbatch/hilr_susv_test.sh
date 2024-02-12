CKPT=/mnt/desa_data/checkpoints/finetune_llama_ehilr
HF=/mnt/desa_data/hfized/finetune_llama_ehilr
LOG=/mnt/desa_data/logs/finetune_llama_ehilr
HESS=/mnt/desa_data/hessians

mkdir $CKPT
mkdir $HF
mkdir $LOG

python finetune_susv_adam.py --save_path $CKPT/2_7b_2bit --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/2_7b_2bit 2>&1

CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_7b_2bit --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1

python tune_susv_lmhead.py --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 3072 --ft_update_freq 2 --ckpt_path $CKPT/2_7b_2bit >> $LOG/2_7b_2bit 2>&1


python finetune_susv_adam.py --save_path $CKPT/2_13b_2bit --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf  --hessian_path $HESS/llama2_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/2_13b_2bit 2>&1

CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_13b_2bit --hf_output_path $HF/2_13b_2bit >> $LOG/2_13b_2bit 2>&1

python tune_susv_lmhead.py --base_model meta-llama/Llama-2-13b-hf --hf_path $HF/2_13b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 3072 --ft_update_freq 2 --ckpt_path $CKPT/2_13b_2bit >> $LOG/2_13b_2bit 2>&1


python finetune_susv_adam.py --save_path $CKPT/2_70b_2bit --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf  --hessian_path $HESS/llama2_70b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/2_70b_2bit 2>&1

CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_2bit --hf_output_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1

python tune_susv_lmhead.py --base_model meta-llama/Llama-2-70b-hf --hf_path $HF/2_70b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 3072 --ft_update_freq 2 --ckpt_path $CKPT/2_70b_2bit >> $LOG/2_70b_2bit 2>&1

CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_2bit --hf_output_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &

CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_13b_2bit --hf_output_path $HF/2_13b_2bit >> $LOG/2_13b_2bit 2>&1 &

CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/2_7b_2bit --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_13b_2bit >> $LOG/2_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_2bit >> $LOG/2_13b_2bit 2>&1 &

CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 &

CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &
wait




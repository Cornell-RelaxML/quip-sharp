CKPT=/mnt/desa_data/checkpoints/fttest
HF=/mnt/desa_data/hfized/fttest
LOG=/mnt/desa_data/logs/fttest
HESS=/mnt/desa_data/hessians

mkdir $CKPT
mkdir $HF
mkdir $LOG


#python quantize_finetune_llama.py --save_path $CKPT/2_70b_2bit --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf  --hessian_path $HESS/llama2_70b_6144/ --devset_size 384 --ft_valid_size 128 >> $LOG/2_70b_2bit 2>&1

#CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_70b_2bit --hf_output_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1

python finetune_e2e_llama.py --base_model meta-llama/Llama-2-70b-hf --hf_path $HF/2_70b_2bit --devset_size 384 --ft_valid_size 128 --ft_epochs 8  --ft_bs 1 --ctx_size 4096 --ft_update_freq 2 --ckpt_path $CKPT/2_70b_2bit --ft_grad_ckpt --ft_train_mode >> $LOG/2_70b_2bit 2>&1

CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_70b_2bit --hf_output_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 

CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &   
wait




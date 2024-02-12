CKPT=/mnt/desa_data/checkpoints/finetune_albert
HF=/mnt/desa_data/hfized/finetune_albert
LOG=/mnt/desa_data/logs/finetune_albert
HESS=/mnt/desa_data/hessians

python finetune_susv.py --save_path $CKPT/2_70b_susv --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf  --hessian_path $HESS/llama2_70b_6144/ --devset_size 640 --ft_valid_size 128 --ft_epochs 5 --ft_bs 4 --ft_update_freq 2 >> $LOG/2_70b_susv 2>&1
CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_susv --hf_output_path $HF/2_70b_susv >> $LOG/2_70b_susv 2>&1
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_70b_susv >> $LOG/2_70b_susv 2>&1

python finetune_susv.py --save_path $CKPT/2_13b_susv --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-13b-hf  --hessian_path $HESS/llama2_13b_6144/ --devset_size 640 --ft_valid_size 128 --ft_epochs 5 >> $LOG/2_13b_susv 2>&1
CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_13b_susv --hf_output_path $HF/2_13b_susv >> $LOG/2_13b_susv 2>&1
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_13b_susv >> $LOG/2_13b_susv 2>&1

python finetune_susv.py --save_path $CKPT/2_7b_susv --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144/ --devset_size 640 --ft_valid_size 128 --ft_epochs 5 >> $LOG/2_7b_susv 2>&1
CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_7b_susv --hf_output_path $HF/2_7b_susv >> $LOG/2_7b_susv 2>&1
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_7b_susv >> $LOG/2_7b_susv 2>&1





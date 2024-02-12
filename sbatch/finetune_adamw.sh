CKPT=/mnt/desa_data/checkpoints/finetune_llama_adamw
HF=/mnt/desa_data/hfized/finetune_llama_adamw
LOG=/mnt/desa_data/logs/finetune_llama_adamw
HESS=/mnt/desa_data/hessians

python finetune_susv_adam.py --save_path $CKPT/2_70b_2bit --codebook E8P12  --scale_override 0.9 --base_model meta-llama/Llama-2-70b-hf  --hessian_path $HESS/llama2_70b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 8 --ft_lr 0.000025 --ft_opt adamw --ft_bs 4 >> $LOG/2_70b_2bit 2>&1

python finetune_susv_adam.py --save_path $CKPT/2_70b_3bit --codebook E8P12RVQ3B --base_model meta-llama/Llama-2-70b-hf  --hessian_path $HESS/llama2_70b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 8 --ft_lr 0.000025 --ft_opt adamw --ft_bs 4 >> $LOG/2_70b_3bit 2>&1

python finetune_susv_adam.py --save_path $CKPT/2_70b_4bit --codebook E8P12RVQ4B --scale_override 0.9 --resid_scale_override 3.6 --base_model meta-llama/Llama-2-70b-hf  --hessian_path $HESS/llama2_70b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 8 --ft_lr 0.000025 --ft_opt adamw --ft_bs 4 >> $LOG/2_70b_4bit 2>&1


python finetune_susv_adam.py --save_path $CKPT/2_13b_4bit --codebook E8P12RVQ4B --scale_override 0.9 --resid_scale_override 3.45 --base_model meta-llama/Llama-2-13b-hf  --hessian_path $HESS/llama2_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 8 --ft_lr 0.000025 --ft_opt adamw --ft_bs 4 >> $LOG/2_13b_4bit 2>&1

python finetune_susv_adam.py --save_path $CKPT/2_7b_4bit --codebook E8P12RVQ4B --scale_override 0.9 --resid_scale_override 3.6 --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 8 --ft_lr 0.000025 --ft_opt adamw  --ft_bs 4 >> $LOG/2_7b_4bit 2>&1



CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_70b_3bit --hf_output_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_2bit --hf_output_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_70b_4bit --hf_output_path $HF/2_70b_4bit >> $LOG/2_70b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/2_13b_4bit --hf_output_path $HF/2_13b_3bit >> $LOG/2_13b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/2_7b_4bit --hf_output_path $HF/2_13b_4bit >> $LOG/2_7b_4bit 2>&1 &

wait

python tune_susv_lmhead.py --base_model meta-llama/Llama-2-70b-hf --hf_path $HF/2_70b_3bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.000001 --ft_bs 1 --ctx_size 3072 --ft_opt adamw --ckpt_path $CKPT/2_70b_3bit >> $LOG/2_70b_3bit 2>&1
python tune_susv_lmhead.py --base_model meta-llama/Llama-2-70b-hf --hf_path $HF/2_70b_4bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.000001 --ft_bs 1 --ctx_size 3072 --ft_opt adamw --ckpt_path $CKPT/2_70b_4bit >> $LOG/2_70b_4bit 2>&1
python tune_susv_lmhead.py --base_model meta-llama/Llama-2-70b-hf --hf_path $HF/2_70b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.000001 --ft_bs 1 --ctx_size 3072 --ft_opt adamw --ckpt_path $CKPT/2_70b_2bit >> $LOG/2_70b_2bit 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 python tune_susv_lmhead.py --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_4bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.000001   --ft_opt adamw --ft_bs 1 --ctx_size 4096 --ckpt_path $CKPT/2_7b_4bit >> $LOG/2_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model meta-llama/Llama-2-13b-hf --hf_path $HF/2_13b_4bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.000001 --ft_opt adamw --ft_bs 1 --ctx_size 4096  --ckpt_path $CKPT/2_13b_4bit >> $LOG/2_13b_4bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_70b_3bit --hf_output_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/2_70b_2bit --hf_output_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/2_70b_4bit --hf_output_path $HF/2_70b_4bit >> $LOG/2_70b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/2_13b_4bit --hf_output_path $HF/2_13b_4bit >> $LOG/2_13b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/2_7b_4bit --hf_output_path $HF/2_7b_4bit >> $LOG/2_7b_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path $HF/2_70b_4bit >> $LOG/2_70b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --hf_path $HF/2_13b_4bit >> $LOG/2_13b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --hf_path $HF/2_7b_4bit >> $LOG/2_7b_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_4bit >> $LOG/2_70b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &   
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_4bit >> $LOG/2_13b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_4bit >> $LOG/2_7b_4bit 2>&1 &     
wait



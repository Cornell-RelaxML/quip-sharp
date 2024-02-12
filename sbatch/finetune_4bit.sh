CKPT=/mnt/desa_data/checkpoints/finetune_llama
HF=/mnt/desa_data/hfized/finetune_llama
LOG=/mnt/desa_data/logs/finetune_llama
HESS=/mnt/desa_data/hessians
'''
# llama 2 4 bit_scale

python finetune_susv_adam.py --save_path $CKPT/2_70b_4bit_scale --scale_override 0.9 --resid_scale_override 3.6 --codebook E8P12RVQ4B --base_model meta-llama/Llama-2-70b-hf  --hessian_path $HESS/llama2_70b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/2_70b_4bit_scale 2>&1

python finetune_susv_adam.py --save_path $CKPT/2_13b_4bit_scale --scale_override 0.9 --resid_scale_override 3.45 --codebook E8P12RVQ4B --base_model meta-llama/Llama-2-13b-hf  --hessian_path $HESS/llama2_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 >> $LOG/2_13b_4bit_scale 2>&1

python finetune_susv_adam.py --save_path $CKPT/2_7b_4bit_scale --scale_override 0.9 --resid_scale_override 3.6 --codebook E8P12RVQ4B --base_model meta-llama/Llama-2-7b-hf  --hessian_path $HESS/llama2_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 >> $LOG/2_7b_4bit_scale 2>&1

# llama 1 4 bit_scale

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_65b_4bit_scale --scale_override 0.9 --resid_scale_override 3.45 --codebook E8P12RVQ4B --base_model relaxml/Llama-1-65b-hf  --hessian_path $HESS/llama1_65b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/1_65b_4bit_scale 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_30b_4bit_scale --scale_override 0.9 --resid_scale_override 3.45 --codebook E8P12RVQ4B --base_model relaxml/Llama-1-30b-hf  --hessian_path $HESS/llama1_30b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/1_30b_4bit_scale 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_13b_4bit_scale --scale_override 0.9 --resid_scale_override 3.6 --codebook E8P12RVQ4B --base_model relaxml/Llama-1-13b-hf  --hessian_path $HESS/llama1_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 >> $LOG/1_13b_4bit_scale 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_7b_4bit_scale --scale_override 0.85 --resid_scale_override 3.45 --codebook E8P12RVQ4B --base_model relaxml/Llama-1-7b-hf  --hessian_path $HESS/llama1_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 >> $LOG/1_7b_4bit_scale 2>&1

# llama 1 3 bit_scale

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_65b_3bit_scale --scale_override 0.93 --resid_scale_override 1.99 --codebook E8P12RVQ3B --base_model relaxml/Llama-1-65b-hf  --hessian_path $HESS/llama1_65b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/1_65b_3bit_scale 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_30b_3bit_scale --scale_override 0.93 --resid_scale_override 2.04 --codebook E8P12RVQ3B --base_model relaxml/Llama-1-30b-hf  --hessian_path $HESS/llama1_30b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 --ft_bs 4 --ft_update_freq 2 >> $LOG/1_30b_3bit_scale 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_13b_3bit_scale --scale_override 0.98 --resid_scale_override 2.09 --codebook E8P12RVQ3B --base_model relaxml/Llama-1-13b-hf  --hessian_path $HESS/llama1_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 >> $LOG/1_13b_3bit_scale 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_7b_3bit_scale --scale_override 0.93 --resid_scale_override 2.09 --codebook E8P12RVQ3B --base_model relaxml/Llama-1-7b-hf  --hessian_path $HESS/llama1_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5 --ft_lr 0.00005 >> $LOG/1_7b_3bit_scale 2>&1


CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_4bit_scale --hf_output_path $HF/2_70b_4bit_scale >> $LOG/2_70b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_13b_4bit_scale --hf_output_path $HF/2_13b_4bit_scale >> $LOG/2_13b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_7b_4bit_scale  --hf_output_path $HF/2_7b_4bit_scale  >> $LOG/2_7b_4bit_scale  2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_65b_4bit_scale --hf_output_path $HF/1_65b_4bit_scale >> $LOG/1_65b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/1_30b_4bit_scale --hf_output_path $HF/1_30b_4bit_scale >> $LOG/1_30b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/1_13b_4bit_scale --hf_output_path $HF/1_13b_4bit_scale >> $LOG/1_13b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/1_7b_4bit_scale  --hf_output_path $HF/1_7b_4bit_scale  >> $LOG/1_7b_4bit_scale  2>&1 &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/1_65b_3bit_scale --hf_output_path $HF/1_65b_3bit_scale >> $LOG/1_65b_3bit_scale 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/1_30b_3bit_scale --hf_output_path $HF/1_30b_3bit_scale >> $LOG/1_30b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/1_13b_3bit_scale --hf_output_path $HF/1_13b_3bit_scale >> $LOG/1_13b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/1_7b_3bit_scale  --hf_output_path $HF/1_7b_3bit_scale  >> $LOG/1_7b_3bit_scale  2>&1 &

wait

# tune llama 2 4 bit_scale
python tune_susv_lmhead.py --base_model meta-llama/Llama-2-70b-hf --hf_path $HF/2_70b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 3072 --ft_update_freq 2 --ckpt_path $CKPT/2_70b_4bit_scale >> $LOG/2_70b_4bit_scale 2>&1
#CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model meta-llama/Llama-2-13b-hf --hf_path $HF/2_13b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 4096 --ft_update_freq 2 --ckpt_path $CKPT/2_13b_4bit_scale >> $LOG/2_13b_4bit_scale 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3 python tune_susv_lmhead.py --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 4096 --ft_update_freq 2 --ckpt_path $CKPT/2_7b_4bit_scale >> $LOG/2_7b_4bit_scale 2>&1 &
wait


python tune_susv_lmhead.py --base_model relaxml/Llama-1-65b-hf --hf_path $HF/1_65b_3bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_65b_3bit_scale >> $LOG/1_65b_3bit_scale 2>&1
python tune_susv_lmhead.py --base_model relaxml/Llama-1-65b-hf --hf_path $HF/1_65b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_65b_4bit_scale >> $LOG/1_65b_4bit_scale 2>&1
python tune_susv_lmhead.py --base_model relaxml/Llama-1-30b-hf --hf_path $HF/1_30b_3bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_30b_3bit_scale >> $LOG/1_30b_3bit_scale 2>&1
#python tune_susv_lmhead.py --base_model relaxml/Llama-1-30b-hf --hf_path $HF/1_30b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_30b_4bit_scale >> $LOG/1_30b_4bit_scale 2>&1
#CUDA_VISIBLE_DEVICES=0,1,2,3 python tune_susv_lmhead.py --base_model relaxml/Llama-1-13b-hf --hf_path $HF/1_13b_3bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_13b_3bit_scale >> $LOG/1_13b_3bit_scale 2>&1 &
#CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-13b-hf --hf_path $HF/1_13b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_13b_4bit_scale >> $LOG/1_13b_4bit_scale 2>&1 &
wait
#CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-7b-hf --hf_path $HF/1_7b_3bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_7b_3bit_scale >> $LOG/1_7b_3bit_scale 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3 python tune_susv_lmhead.py --base_model relaxml/Llama-1-7b-hf --hf_path $HF/1_7b_4bit_scale --devset_size 240 --ft_valid_size 40 --ft_epochs 8 --ft_lr 0.00001 --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_7b_4bit_scale >> $LOG/1_7b_4bit_scale 2>&1 &
wait




CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/2_70b_4bit_scale --hf_output_path $HF/2_70b_4bit_scale >> $LOG/2_70b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/2_13b_4bit_scale --hf_output_path $HF/2_13b_4bit_scale >> $LOG/2_13b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/2_7b_4bit_scale  --hf_output_path $HF/2_7b_4bit_scale  >> $LOG/2_7b_4bit_scale  2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_65b_4bit_scale --hf_output_path $HF/1_65b_4bit_scale >> $LOG/1_65b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/1_30b_4bit_scale --hf_output_path $HF/1_30b_4bit_scale >> $LOG/1_30b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/1_13b_4bit_scale --hf_output_path $HF/1_13b_4bit_scale >> $LOG/1_13b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/1_7b_4bit_scale  --hf_output_path $HF/1_7b_4bit_scale  >> $LOG/1_7b_4bit_scale  2>&1 &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/1_65b_3bit_scale --hf_output_path $HF/1_65b_3bit_scale >> $LOG/1_65b_3bit_scale 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/1_30b_3bit_scale --hf_output_path $HF/1_30b_3bit_scale >> $LOG/1_30b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/1_13b_3bit_scale --hf_output_path $HF/1_13b_3bit_scale >> $LOG/1_13b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/1_7b_3bit_scale  --hf_output_path $HF/1_7b_3bit_scale  >> $LOG/1_7b_3bit_scale  2>&1 &

wait

#CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_70b_4bit_scale >> $LOG/2_70b_4bit_scale 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/2_13b_4bit_scale >> $LOG/2_13b_4bit_scale 2>&1 &
#CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/2_7b_4bit_scale  >> $LOG/2_7b_4bit_scale  2>&1 &
wait
'''
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path $HF/1_65b_4bit_scale --seqlen 2048 >> $LOG/1_65b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/1_30b_4bit_scale --seqlen 2048 >> $LOG/1_30b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/1_13b_4bit_scale --seqlen 2048 >> $LOG/1_13b_4bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --hf_path $HF/1_7b_4bit_scale  --seqlen 2048 >> $LOG/1_7b_4bit_scale  2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --hf_path $HF/1_65b_3bit_scale --seqlen 2048 >> $LOG/1_65b_3bit_scale 2>&1 &					                                                    
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/1_30b_3bit_scale --seqlen 2048 >> $LOG/1_30b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/1_13b_3bit_scale --seqlen 2048 >> $LOG/1_13b_3bit_scale 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/1_7b_3bit_scale  --seqlen 2048 >> $LOG/1_7b_3bit_scale  2>&1 &

wait
'''
CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/2_70b_4bit_scale >> $LOG/2_70b_4bit_scale 2>&1 & 
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/2_13b_4bit_scale >> $LOG/2_13b_4bit_scale 2>&1 & 
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/2_7b_4bit_scale  >> $LOG/2_7b_4bit_scale  2>&1 & 
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_65b_4bit_scale >> $LOG/1_65b_4bit_scale 2>&1 & 
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_30b_4bit_scale >> $LOG/1_30b_4bit_scale 2>&1 & 
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_13b_4bit_scale >> $LOG/1_13b_4bit_scale 2>&1 & 
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_7b_4bit_scale  >> $LOG/1_7b_4bit_scale  2>&1 & 
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_65b_3bit_scale >> $LOG/1_65b_3bit_scale 2>&1 & 
														                                                       
wait														                                                       
														                                                       
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_30b_3bit_scale >> $LOG/1_30b_3bit_scale 2>&1 & 
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_13b_3bit_scale >> $LOG/1_13b_3bit_scale 2>&1 & 
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4  --hf_path $HF/1_7b_3bit_scale  >> $LOG/1_7b_3bit_scale  2>&1 & 
wait


'''

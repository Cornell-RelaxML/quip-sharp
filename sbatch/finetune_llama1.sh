CKPT=/mnt/desa_data/checkpoints/icml_llama
HF=/mnt/desa_data/hfized/icml_llama
LOG=/mnt/desa_data/logs/icml_llama
HESS=/mnt/desa_data/hessians

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_65b_3bit --codebook E8P12RVQ3B --base_model relaxml/Llama-1-65b-hf  --hessian_path $HESS/llama1_65b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  --ft_bs 4 --ft_update_freq 2 >> $LOG/1_65b_3bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_30b_3bit --codebook E8P12RVQ3B --base_model relaxml/Llama-1-30b-hf  --hessian_path $HESS/llama1_30b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  --ft_bs 4 --ft_update_freq 2 >> $LOG/1_30b_3bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_13b_3bit --codebook E8P12RVQ3B --base_model relaxml/Llama-1-13b-hf  --hessian_path $HESS/llama1_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  >> $LOG/1_13b_3bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_7b_3bit --codebook E8P12RVQ3B --base_model relaxml/Llama-1-7b-hf  --hessian_path $HESS/llama1_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  >> $LOG/1_7b_3bit 2>&1


python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_65b_4bit --codebook E8P12RVQ4B --base_model relaxml/Llama-1-65b-hf  --hessian_path $HESS/llama1_65b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  --ft_bs 4 --ft_update_freq 2 >> $LOG/1_65b_4bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_30b_4bit --codebook E8P12RVQ4B --base_model relaxml/Llama-1-30b-hf  --hessian_path $HESS/llama1_30b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  --ft_bs 4 --ft_update_freq 2 >> $LOG/1_30b_4bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_13b_4bit --codebook E8P12RVQ4B --base_model relaxml/Llama-1-13b-hf  --hessian_path $HESS/llama1_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  >> $LOG/1_13b_4bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_7b_4bit --codebook E8P12RVQ4B --base_model relaxml/Llama-1-7b-hf  --hessian_path $HESS/llama1_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  >> $LOG/1_7b_4bit 2>&1


python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_65b_2bit --codebook E8P12  --scale_override 0.9 --base_model relaxml/Llama-1-65b-hf  --hessian_path $HESS/llama1_65b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  --ft_bs 4 --ft_update_freq 2 >> $LOG/1_65b_2bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_30b_2bit --codebook E8P12  --scale_override 0.9 --base_model relaxml/Llama-1-30b-hf  --hessian_path $HESS/llama1_30b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  --ft_bs 4 --ft_update_freq 2 >> $LOG/1_30b_2bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_13b_2bit --codebook E8P12  --scale_override 0.9 --base_model relaxml/Llama-1-13b-hf  --hessian_path $HESS/llama1_13b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  >> $LOG/1_13b_2bit 2>&1

python finetune_susv_adam.py --ctx_size 2048 --save_path $CKPT/1_7b_2bit --codebook E8P12  --scale_override 0.9 --base_model relaxml/Llama-1-7b-hf  --hessian_path $HESS/llama1_7b_6144/ --devset_size 384 --ft_valid_size 128 --ft_epochs 5  >> $LOG/1_7b_2bit 2>&1

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/1_65b_3bit --hf_output_path $HF/1_65b_3bit >> $LOG/1_65b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/1_65b_2bit --hf_output_path $HF/1_65b_2bit >> $LOG/1_65b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/1_65b_4bit --hf_output_path $HF/1_65b_4bit >> $LOG/1_65b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_30b_3bit --hf_output_path $HF/1_30b_3bit >> $LOG/1_30b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/1_30b_2bit --hf_output_path $HF/1_30b_2bit >> $LOG/1_30b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/1_30b_4bit --hf_output_path $HF/1_30b_4bit >> $LOG/1_30b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/1_13b_3bit --hf_output_path $HF/1_13b_3bit >> $LOG/1_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/1_13b_4bit --hf_output_path $HF/1_13b_4bit >> $LOG/1_13b_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/1_13b_2bit --hf_output_path $HF/1_13b_2bit >> $LOG/1_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/1_7b_3bit --hf_output_path $HF/1_7b_3bit >> $LOG/1_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/1_7b_4bit --hf_output_path $HF/1_7b_4bit >> $LOG/1_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_7b_2bit --hf_output_path $HF/1_7b_2bit >> $LOG/1_7b_2bit 2>&1 &

wait


python tune_susv_lmhead.py --base_model relaxml/Llama-1-65b-hf --hf_path $HF/1_65b_3bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_65b_3bit >> $LOG/1_65b_3bit 2>&1
python tune_susv_lmhead.py --base_model relaxml/Llama-1-65b-hf --hf_path $HF/1_65b_4bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_65b_4bit >> $LOG/1_65b_4bit 2>&1
python tune_susv_lmhead.py --base_model relaxml/Llama-1-65b-hf --hf_path $HF/1_65b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_65b_2bit >> $LOG/1_65b_2bit 2>&1

python tune_susv_lmhead.py --base_model relaxml/Llama-1-30b-hf --hf_path $HF/1_30b_3bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_30b_3bit >> $LOG/1_30b_3bit 2>&1
python tune_susv_lmhead.py --base_model relaxml/Llama-1-30b-hf --hf_path $HF/1_30b_4bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_30b_4bit >> $LOG/1_30b_4bit 2>&1
python tune_susv_lmhead.py --base_model relaxml/Llama-1-30b-hf --hf_path $HF/1_30b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_30b_2bit >> $LOG/1_30b_2bit 2>&1


CUDA_VISIBLE_DEVICES=0,1,2,3 python tune_susv_lmhead.py --base_model relaxml/Llama-1-13b-hf --hf_path $HF/1_13b_3bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_13b_3bit >> $LOG/1_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-13b-hf --hf_path $HF/1_13b_4bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_13b_4bit >> $LOG/1_13b_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0,1,2,3 python tune_susv_lmhead.py --base_model relaxml/Llama-1-13b-hf --hf_path $HF/1_13b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_13b_2bit >> $LOG/1_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-7b-hf --hf_path $HF/1_7b_3bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_7b_3bit >> $LOG/1_7b_3bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0,1,2,3 python tune_susv_lmhead.py --base_model relaxml/Llama-1-7b-hf --hf_path $HF/1_7b_4bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_7b_4bit >> $LOG/1_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 python tune_susv_lmhead.py --base_model relaxml/Llama-1-7b-hf --hf_path $HF/1_7b_2bit --devset_size 240 --ft_valid_size 40 --ft_epochs 8  --ft_bs 1 --ctx_size 2048 --ft_update_freq 2 --ckpt_path $CKPT/1_7b_2bit >> $LOG/1_7b_2bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/1_65b_3bit --hf_output_path $HF/1_65b_3bit >> $LOG/1_65b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/1_65b_2bit --hf_output_path $HF/1_65b_2bit >> $LOG/1_65b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/1_65b_4bit --hf_output_path $HF/1_65b_4bit >> $LOG/1_65b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_30b_3bit --hf_output_path $HF/1_30b_3bit >> $LOG/1_30b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python hfize_llama.py --quantized_path $CKPT/1_30b_2bit --hf_output_path $HF/1_30b_2bit >> $LOG/1_30b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python hfize_llama.py --quantized_path $CKPT/1_30b_4bit --hf_output_path $HF/1_30b_4bit >> $LOG/1_30b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python hfize_llama.py --quantized_path $CKPT/1_13b_3bit --hf_output_path $HF/1_13b_3bit >> $LOG/1_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python hfize_llama.py --quantized_path $CKPT/1_13b_4bit --hf_output_path $HF/1_13b_4bit >> $LOG/1_13b_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path $CKPT/1_13b_2bit --hf_output_path $HF/1_13b_2bit >> $LOG/1_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python hfize_llama.py --quantized_path $CKPT/1_7b_3bit --hf_output_path $HF/1_7b_3bit >> $LOG/1_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python hfize_llama.py --quantized_path $CKPT/1_7b_4bit --hf_output_path $HF/1_7b_4bit >> $LOG/1_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python hfize_llama.py --quantized_path $CKPT/1_7b_2bit --hf_output_path $HF/1_7b_2bit >> $LOG/1_7b_2bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_65b_3bit >> $LOG/1_65b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_65b_4bit >> $LOG/1_65b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_65b_2bit >> $LOG/1_65b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_30b_3bit >> $LOG/1_30b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_30b_4bit >> $LOG/1_30b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_30b_2bit >> $LOG/1_30b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_13b_3bit >> $LOG/1_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_13b_4bit >> $LOG/1_13b_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_13b_2bit >> $LOG/1_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_7b_3bit >> $LOG/1_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_7b_4bit >> $LOG/1_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_7b_2bit >> $LOG/1_7b_2bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_3bit >> $LOG/1_65b_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_4bit >> $LOG/1_65b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_2bit >> $LOG/1_65b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_3bit >> $LOG/1_30b_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_4bit >> $LOG/1_30b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_2bit >> $LOG/1_30b_2bit 2>&1 &   
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_3bit >> $LOG/1_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_4bit >> $LOG/1_13b_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_2bit >> $LOG/1_13b_2bit 2>&1 &   
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_3bit >> $LOG/1_7b_3bit 2>&1 &     
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_4bit >> $LOG/1_7b_4bit 2>&1 &     
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_2bit >> $LOG/1_7b_2bit 2>&1 &     
wait



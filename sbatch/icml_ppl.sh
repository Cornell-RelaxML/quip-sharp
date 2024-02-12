HF=/mnt/desa_data/hfized
LOG=/mnt/desa_data/logs/icml_ppl

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/new_e8p/2_70b_e8p_2bit --seqlen 2048 >> $LOG/2_70b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/rvq/2_70b_e8prvq_3bit --seqlen 2048 >> $LOG/2_70b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/rvq/2_70b_e8prvq_4bit --seqlen 2048 >> $LOG/2_70b_e8prvq_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path $HF/new_e8p/2_7b_e8p_2bit --seqlen 2048 >> $LOG/2_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/new_e8p/2_13b_e8p_2bit --seqlen 2048 >> $LOG/2_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/rvq/2_13b_e8prvq_3bit --seqlen 2048 >> $LOG/2_13b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --hf_path $HF/rvq/2_13b_e8prvq_4bit --seqlen 2048 >> $LOG/2_13b_e8prvq_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --hf_path meta-llama/Llama-2-13b-hf --seqlen 2048 >> $LOG/2_13b_fp16 2>&1 & 

wait

CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/rvq/2_7b_e8prvq_3bit --seqlen 2048 >> $LOG/2_7b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/rvq/2_7b_e8prvq_4bit --seqlen 2048 >> $LOG/2_7b_e8prvq_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path meta-llama/Llama-2-7b-hf --seqlen 2048 >> $LOG/2_7b_fp16 2>&1 & 
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/new_e8p/1_65b_e8p_2bit --seqlen 2048 >> $LOG/1_65b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/rvq/1_65b_e8prvq_3bit --seqlen 2048 >> $LOG/1_65b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/rvq/1_65b_e8prvq_4bit --seqlen 2048 >> $LOG/1_65b_e8prvq_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 python eval_ppl.py --hf_path meta-llama/Llama-2-70b-hf --seqlen 2048 >> $LOG/2_70b_fp16 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/new_e8p/1_30b_e8p_2bit --seqlen 2048 >> $LOG/1_30b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/rvq/1_30b_e8prvq_3bit --seqlen 2048 >> $LOG/1_30b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/rvq/1_30b_e8prvq_4bit --seqlen 2048 >> $LOG/1_30b_e8prvq_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path relaxml/Llama-1-30b-hf --seqlen 2048 >> $LOG/1_30b_fp16 2>&1 & 
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/new_e8p/1_13b_e8p_2bit --seqlen 2048 >> $LOG/1_13b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/rvq/1_13b_e8prvq_3bit --seqlen 2048 >> $LOG/1_13b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --hf_path $HF/rvq/1_13b_e8prvq_4bit --seqlen 2048 >> $LOG/1_13b_e8prvq_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --hf_path relaxml/Llama-1-13b-hf --seqlen 2048 >> $LOG/1_13b_fp16 2>&1 & 

wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/new_e8p/1_7b_e8p_2bit --seqlen 2048 >> $LOG/1_7b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/rvq/1_7b_e8prvq_3bit --seqlen 2048 >> $LOG/1_7b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/rvq/1_7b_e8prvq_4bit --seqlen 2048 >> $LOG/1_7b_e8prvq_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path relaxml/Llama-1-7b-hf --seqlen 2048 >> $LOG/1_7b_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/falcon_180b_e8p_2bit --seqlen 2048 --no_use_cuda_graph >> $LOG/falcon_180b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/falcon_180b_e8prvq_3bit --seqlen 2048 --no_use_cuda_graph >> $LOG/falcon_180b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 python eval_ppl.py --hf_path $HF/falcon_180b_e8prvq_4bit --seqlen 2048 --no_use_cuda_graph >> $LOG/falcon_180b_e8prvq_4bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python eval_ppl.py --hf_path tiiuae/falcon-180B --seqlen 2048 --no_use_cuda_graph >> $LOG/falcon_180b_fp16 2>&1 &
CUDA_VISIBLE_DEVICES=5,6 python eval_ppl.py --hf_path relaxml/Llama-1-65b-hf --seqlen 2048 >> $LOG/1_65b_fp16 2>&1 & 

wait

HF=/mnt/desa_data/hfized
LOG=/mnt/desa_data/logs

CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/falcon_180b_e8p_2bit/ >> $LOG/falcon_180b_e8p_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/falcon_180b_e8prvq_3bit/ >> $LOG/falcon_180b_e8prvq_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/falcon_180b_e8prvq_4bit/ >> $LOG/falcon_180b_e8prvq_4bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path tiiuae/falcon-180B >> $LOG/falcon_180b_fp16 2>&1

HF=/mnt/desa_data/hfized/icml_llama
LOG=/mnt/desa_data/logs/icml_llama_eval

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_65b_3bit >> $LOG/1_65b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_65b_4bit >> $LOG/1_65b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_65b_2bit >> $LOG/1_65b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/2_70b_4bit >> $LOG/2_70b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --hf_path $HF/2_70b_chat_3bit >> $LOG/2_70b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --hf_path $HF/2_70b_chat_4bit >> $LOG/2_70b_chat_4bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_70b_chat_2bit >> $LOG/2_70b_chat_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_30b_3bit >> $LOG/1_30b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_30b_4bit >> $LOG/1_30b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_30b_2bit >> $LOG/1_30b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_13b_3bit >> $LOG/1_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_13b_4bit >> $LOG/1_13b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_13b_2bit >> $LOG/1_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --hf_path $HF/2_13b_3bit >> $LOG/2_13b_3bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_13b_4bit >> $LOG/2_13b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/2_13b_2bit >> $LOG/2_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/2_13b_chat_3bit >> $LOG/2_13b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path $HF/2_13b_chat_4bit >> $LOG/2_13b_chat_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/2_13b_chat_2bit >> $LOG/2_13b_chat_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_7b_3bit >> $LOG/1_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_7b_4bit >> $LOG/1_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_ppl.py --seqlen 2048 --hf_path $HF/1_7b_2bit >> $LOG/1_7b_2bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path $HF/2_7b_3bit >> $LOG/2_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --hf_path $HF/2_7b_4bit >> $LOG/2_7b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --hf_path $HF/2_7b_chat_3bit >> $LOG/2_7b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_ppl.py --hf_path $HF/2_7b_chat_4bit >> $LOG/2_7b_chat_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --hf_path $HF/2_7b_chat_2bit >> $LOG/2_7b_chat_2bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_3bit >> $LOG/1_65b_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_4bit >> $LOG/1_65b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_65b_2bit >> $LOG/1_65b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_4bit >> $LOG/2_70b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_2bit >> $LOG/2_70b_2bit 2>&1 &   
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_chat_3bit >> $LOG/2_70b_chat_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_chat_4bit >> $LOG/2_70b_chat_4bit 2>&1 &

CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_70b_chat_2bit >> $LOG/2_70b_chat_2bit 2>&1 &   
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_3bit >> $LOG/1_30b_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_4bit >> $LOG/1_30b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_30b_2bit >> $LOG/1_30b_2bit 2>&1 &   
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_3bit >> $LOG/1_13b_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_4bit >> $LOG/1_13b_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_13b_2bit >> $LOG/1_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_3bit >> $LOG/2_13b_3bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_4bit >> $LOG/2_13b_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_2bit >> $LOG/2_13b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_chat_3bit >> $LOG/2_13b_chat_3bit 2>&1 &   
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_chat_4bit >> $LOG/2_13b_chat_4bit 2>&1 &   
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_13b_chat_2bit >> $LOG/2_13b_chat_2bit 2>&1 &   
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_3bit >> $LOG/1_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_4bit >> $LOG/1_7b_4bit 2>&1 &     
CUDA_VISIBLE_DEVICES=7 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/1_7b_2bit >> $LOG/1_7b_2bit 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_3bit >> $LOG/2_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_4bit >> $LOG/2_7b_4bit 2>&1 &     
CUDA_VISIBLE_DEVICES=2 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_chat_3bit >> $LOG/2_7b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_chat_4bit >> $LOG/2_7b_chat_4bit 2>&1 &     
CUDA_VISIBLE_DEVICES=5 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 4 --hf_path $HF/2_7b_chat_2bit >> $LOG/2_7b_chat_2bit 2>&1 &
wait




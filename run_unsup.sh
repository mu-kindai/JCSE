#export CUDA_VISIBLE_DEVICES=0
#python simcse_to_huggingface.py\
    #--path "/home/chen/SimCSE/my-unsup-simcse-bert-base-jsts"


python train.py \
    --model_name_or_path cl-tohoku/bert-base-japanese\
    --train_file /home/chen/SimCSE/data/wiki_plus_qa_shuffle.txt \
    --output_dir transfer_model/my-unsup-simcse-bert-base-all-len32-batch64-clinic\
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1\
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model jacsts_spearman\
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    "$@"

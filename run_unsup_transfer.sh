python train.py \
    --model_name_or_path cl-tohoku/bert-large-japanese\
    --train_file /home/chen/SimCSE/data/wiki_plus_qa_shuffle.txt \
    --output_dir qa_transfer_model/my-unsup-simcse-bert-large-all-len32-batch64-qa\
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1\
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --save_strategy steps \
    --save_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    "$@"

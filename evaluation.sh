python simcse_to_huggingface.py\
    --path /home/chen/DiffCSE/result/my-unsup-diffcse-bert-base-all-transfer-clinic

python evaluation.py \
    --model_name_or_path /home/chen/DiffCSE/result/my-unsup-diffcse-bert-base-all-transfer-clinic\
    --pooler  avg\
    --task_set transfer \
    --mode test
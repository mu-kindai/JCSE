#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

#NUM_GPU=3

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
#PORT_ID=$(expr $RANDOM + 1024)

# Allow multiple threads
#export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
#python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
python train.py \
    --model_name_or_path /home/chen/SimCSE/my-sup-simcse-bert-base-hard_neg0-batch-512-jacsts-top4\
    --train_file /home/chen/SimCSE/data/nli_for_simcse.csv \
    --output_dir transfer_model/my-sup-simcse-bert-base-hard_neg1-batch-512-jacsts-top4-plusnli\
    --num_train_epochs 2 \
    --per_device_train_batch_size 128\
    --gradient_accumulation_steps 4\
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 125 \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --hard_negative_weight 1 \
    --do_train \
    --do_eval \
    #--fp16 \
    "$@"
#!/bin/bash
#source ~/.bashrc
#hostname
python entry.py \
    --load_dir=/home/syf/dataset/2017/5_bucket \
    --epoch_num=450 \
    --learning_rate=0.0001 \
    --batch_size=4 \
    --regularization=0.0002 \
    --using_model=False \
    --if_test=False \
    --decay_rate=0.2 \
    --decay_frequency=150 \
    --slow_start=False \
    --clip=False \
    --use_l2=True \
    --hard_label=False \
    --distillation=False \
    --model_path='../../saved_model/model_load/' \
    --fold=0

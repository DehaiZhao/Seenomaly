#!/bin/bash
set -e

MODEL_NAME=vaegan
TRAINSET_NAME=rico_gif
#EVALUATIONSET_NAME=python_0_0

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/cheer/Project/Do_Dont/models/${MODEL_NAME}

# Where the dataset is saved to.
TRAINSET_DIR=/home/cheer/Project/Do_Dont/Rico_Data/tf_record/rico_gif
#EVALUATIONSET_DIR=/home/cheer/video_test/corre/data/${EVALUATIONSET_NAME}


python3 main.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --dataset_dir=${TRAINSET_DIR} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=1000000 \
  --batch_size=8 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=300 \
  --max_to_keep=3 \
  --log_every_n_steps=10 \
  --optimizer=adam \
  --weight_decay=0.00004 \

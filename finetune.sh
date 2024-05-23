#!/usr/bin/env bash

source activate <your-conda-env>

dataset=chime4
model_size=large-v3
train_data=<train-kaldi-data-dir>
dev_data=<dev-kaldi-data-dir>

$cmd log/finetune_${dataset}_${model_size}.log \
python finetune.py \
    --MODEL "openai/whisper-${model_size}" \
    --DATASET ${dataset} \
    --TRAIN_DATA ${train_data} \
    --DEV_DATA ${dev_data} \
    --BATCH_SIZE 1 \
    --GRADIENT_ACCUMULATION_STEPS 16 \
    --LEARNING_RATE 1e-5 \
    --EPOCHS 2 \


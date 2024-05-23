#!/usr/bin/env bash

source activate <your-conda-env>

dataset=chime4
model_size=large-v3
checkpoint=<checkpoint-path>
test_data=<test-kaldi-data-dir>

$cmd log/inference_${dataset}_${model_size}.log \
python inference.py \
    --MODEL "openai/whisper-${model_size}" \
    --DATASET ${dataset} \
    --CKPT ${checkpoint} \
    --TEST_DATA ${test_data}


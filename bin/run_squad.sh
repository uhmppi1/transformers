#!/usr/bin/env bash

cd ..   # BASEDIR인 transformers 디렉토리에서 실행한다.
BASEDIR=$(pwd)

MODEL=bert
MODEL_NAME=bert-large-uncased-whole-word-masking
HPARAMS=hparam_for_kakao_shopping_image_only
SQUAD_DIR=$BASEDIR/data/squad
OUTPUT_DIR=$BASEDIR/models/wwm_uncased_finetuned_squad/

#python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py \
python ./examples/run_squad.py \
    --model_type $MODEL \
    --model_name_or_path $MODEL_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3
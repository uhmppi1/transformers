#!/usr/bin/env bash

cd ..   # BASEDIR인 transformers 디렉토리에서 실행한다.
BASEDIR=$(pwd)

MODEL=bert
MODEL_NAME=bert-base-multilingual-cased
KORQUAD_DIR=$BASEDIR/data/korquad_v1
OUTPUT_DIR=$BASEDIR/models/wwm_uncased_finetuned_korquad_v1/

#python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py \
#python ./examples/run_squad.py \
#python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py \
python ./examples/run_squad.py \
    --model_type $MODEL \
    --model_name_or_path $MODEL_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $KORQUAD_DIR/KorQuAD_v1.0_train.json \
    --predict_file $KORQUAD_DIR/KorQuAD_v1.0_dev.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3
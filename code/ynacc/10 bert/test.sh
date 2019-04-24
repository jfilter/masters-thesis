#!/usr/bin/env bash
set -e
set -x

python test.py --cls $1 --task ynacc --bert_model bert-base-cased --data_dir data --output_dir "$2" --do_eval --eval_batch_size 128


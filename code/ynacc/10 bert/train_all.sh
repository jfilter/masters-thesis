#!/usr/bin/env bash
set -e
set -x


for cls in {0..10}
do
	for epochs in {3..10}
	do
		for lr in {6..8}
		do
			output="reply_output_$cls"
			output+="_$epochs"
			output+="_$lr"
			# rmdir $output &&
			python run_classifier.py --cls $cls --task ynacc --bert_model bert-base-cased --data_dir data --output_dir $output --do_train --num_train_epochs $epochs --train_batch_size 64 --learning_rate 5e-$lr  && 
			python run_classifier.py --no_cuda --cls $cls --task ynacc --bert_model bert-base-cased --data_dir data --output_dir $output --do_eval --eval_batch_size 128
		done
	done
done


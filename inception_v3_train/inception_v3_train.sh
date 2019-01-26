#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

# Where the dataset is saved to.
DATASET_DIR=/mnt/retrain/flowers

PRUNING_HPARAMS="begin_pruning_step=1,pruning_frequency=1,end_pruning_step=100,target_sparsity=0.5"
train_image_classify() {
	model_name=$1
	train_dir=$2

	rm -rf $train_dir
	mkdir $train_dir

	python train/train_image_classifier.py \
	  --model_name=${model_name} \
	  --train_dir=${train_dir} \
	  --dataset_name=flowers \
	  --dataset_split_name=train \
	  --dataset_dir=${DATASET_DIR} \
	  --max_number_of_steps=10 \
	  --batch_size=1 \
	  --learning_rate=0.01 \
	  --learning_rate_decay_type=fixed \
	  --save_interval_secs=3600 \
	  --save_summaries_secs=3600 \
	  --log_every_n_steps=100 \
	  --optimizer=rmsprop \
	  --clone_on_cpu=true \
	  --weight_decay=0.00004 \
	  --num_readers=1 \
	  --num_preprocessing_threads=1 \
	  ${PRUNING}
}

evaluate_model(){
	model_name=$1
	train_dir=$2

	python eval_image_classifier.py \
	  --checkpoint_path=${train_dir} \
	  --eval_dir=${train_dir} \
	  --dataset_name=flowers \
	  --dataset_split_name=validation \
	  --dataset_dir=${DATASET_DIR} \
	  --model_name=${model_name}
}

print_help(){
	cmd=$1
	echo $cmd' [prunable_inception_v3 / inception_v3]'
}

if [ $# -eq 1 ];
then
	if [ $1 = 'prunable_inception_v3' ];
	then
		TRAIN_DIR=inception_v3_flowers_checkpoint/pretrained_inception_v3
		PRUNING='--pruning=True --pruning_hparams='${PRUNING_HPARAMS}
		train_image_classify prunable_inception_v3 ${TRAIN_DIR}
	elif [ $1 = 'inception_v3' ];
	then
		TRAIN_DIR=inception_v3_flowers_checkpoint/inception_v3
		PRUNING='--pruning=False'
		train_image_classify inception_v3 ${TRAIN_DIR}
	else
		print_help $0
	fi
else
	print_help $0
fi
#evaluate_model

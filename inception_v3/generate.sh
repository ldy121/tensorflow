#!/bin/sh

python_script=cnn_neural_network.py
input_path=/home/dy121/tensorflow/pretrained_inception_v3/inception_v3.ckpt
input_name=input
output_name=InceptionV3/Predictions/Reshape_1
inception_v3_script=/opt/script/inception_v3.sh

if [ $# -eq 1 ];
then
	. ${inception_v3_script}

	name=$1
	pre_process
	generate_${name}
	post_process
else
	echo $0' [ pretrained_inception_v3 / zero_inception_v3 / prunable_inception_v3 / pretrained_prunable_inception_v3 ]'
fi

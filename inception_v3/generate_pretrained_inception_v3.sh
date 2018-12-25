#!/bin/sh

name=pretrained_inception_v3
input_path=/home/dy121/tensorflow/pretrained_inception_v3/inception_v3.ckpt
output_name=InceptionV3/Predictions/Reshape_1

rm -rf $name
mkdir $name

python generate_pretrained_inception_v3.py $input_path $name/$name

/opt/script/freeze_graph.sh \
	$name/$name.pbtxt \
	$name/$name \
	$output_name

mv model.pb $name/$name.pb 

#!/bin/sh

name=zeros_inception_v3
output_name=InceptionV3/Predictions/Reshape_1

rm -rf $name
mkdir $name

python generate_all_zeros_inception_v3.py $name/$name

/opt/script/freeze_graph.sh \
	$name/$name.pbtxt \
	$name/$name \
	$output_name

mv model.pb $name/$name.pb 

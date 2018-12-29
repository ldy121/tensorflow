#!/bin/sh

name=prunable_inception_v3
output_name=InceptionV3/Predictions/Reshape_1

rm -rf $name
mkdir $name

python generate_prunable_inception_v3.py $name/$name
/opt/script/freeze_pruning_graph.sh \
	$name/ \
	$output_name

mv model.pb $name/$name.pb 

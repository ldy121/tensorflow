#!/bin/sh

name=prunable_inception_v3
output_name=InceptionV3/Predictions/Reshape_1

#rm $name/*

#python generate_prunable_inception_v3.py $name/$name
/opt/script/freeze_graph.sh \
	$name/$name.pbtxt \
	$name/$name \
	$output_name

mv model.pb $name/$name.pb 

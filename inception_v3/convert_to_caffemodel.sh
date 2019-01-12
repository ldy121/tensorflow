#!/bin/sh

shape=299,299,3
output_name=InceptionV3/Predictions/Reshape_1

if [ $# -eq 1 ] ;
then
	/opt/script/tensorflow_to_caffe.sh $1 $shape $output_name
else
	echo $0' [ pb file ]'
fi

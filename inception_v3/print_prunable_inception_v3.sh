#!/bin/sh

python_script=cnn_neural_network.py

if [ $# -eq 1 ];
then
	python $python_script print_prunable_inception_v3 $1 
else
	echo $0' [ checkpoint ]'
fi

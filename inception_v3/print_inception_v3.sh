#!/bin/sh

python_script=cnn_neural_network.py

if [ $# -eq 1 ];
then
	python3 $python_script print_inception_v3 $1 
else
	echo $0' [ checkpoint ]'
fi

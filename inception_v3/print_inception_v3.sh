#!/bin/sh

python_script=neural_network.py

if [ $# -eq 1 ];
then
	python $python_script print_inception_v3 $1 
else
	echo $0' [ checkpoint ]'
fi

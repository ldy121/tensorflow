#!/bin/sh

python_script=neural_network.py

if [ $# -eq 2 ];
then
	python3 ${python_script} compare_inception_v3 $1 $2
else
	echo $0' [ comp src checkpoint1 ] [ comp src checkpoint2 ]'
fi

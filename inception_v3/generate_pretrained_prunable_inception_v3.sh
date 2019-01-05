#!/bin/sh

name=pretrained_prunable_inception_v3
input_path=/home/dy121/tensorflow/pretrained_inception_v3/inception_v3.ckpt
input_name=input
output_name=InceptionV3/Predictions/Reshape_1

rm -rf ${name}
mkdir ${name}

python generate_pretrained_prunable_inception_v3.py ${input_path} ${name}/${name}
/opt/script/freeze_pruning_graph.sh \
	${name}/ \
	${output_name}

/opt/script/optimize_inference.sh \
	model.pb \
	${input_name} \
	${output_name}

mv model.pb ${name}/${name}.pb 
mv opt_model.pb ${name}/opt_${name}.pb

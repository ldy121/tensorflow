#!/bin/sh

name=zeros_inception_v3
output_name=InceptionV3/Predictions/Reshape_1
input_name=input

rm -rf ${name}
mkdir ${name}

python generate_all_zeros_inception_v3.py ${name}/${name}

/opt/script/freeze_graph.sh \
	${name}/${name}.pbtxt \
	${name}/${name} \
	${output_name}

/opt/script/optimize_inference.sh \
	model.pb \
	${input_name} \
	${output_name}

mv model.pb ${name}/${name}.pb 
mv opt_model.pb ${name}/opt_${name}.pb 

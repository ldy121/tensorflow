#!/bin/sh

python generate_all_zeros_inception_v3.py zeros_inception_v3/zeros_inception_v3
/opt/script/freeze_graph.sh \
	zeros_inception_v3/zeros_inception_v3.pbtxt \
	zeros_inception_v3/zeros_inception_v3 \
	InceptionV3/Predictions/Reshape_1

mv model.pb zeros_inception_v3.pb 

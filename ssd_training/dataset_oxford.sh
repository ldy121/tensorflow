#!/bin/sh
DATA_DIR=oxford_dataset
OUTPUT_DIR=oxford_dataset
PYTHONPATH=./:"$PYTHONPATH"
export PYTHONPATH

# If there is import error by proto files
#protoc object_detection/protos/*.proto --python_out=.

python object_detection/dataset_tools/create_pet_tf_record.py \
    --data_dir=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR}
#    --label_map_path=object_detection/data/pet_label_map.pbtxt \

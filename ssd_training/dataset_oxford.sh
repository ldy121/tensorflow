#!/bin/sh
DATA_DIR=/mnt/oxford_pet
OUTPUT_DIR=/mnt/oxford_pet
PYTHONPATH=./:"$PYTHONPATH"
export PYTHONPATH

# If there is import error by proto files
#protoc object_detection/protos/*.proto --python_out=.

python object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR}

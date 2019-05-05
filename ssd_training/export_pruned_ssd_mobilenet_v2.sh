#!/bin/sh
CONFIG_SCRIPT=./ssd_mobilenet_v2_config.sh
OUTPUT_DIR=pruned_ssd_mobilenet_v2_output

. ${CONFIG_SCRIPT}

rm -rf ${OUTPUT_DIR}

python object_detection/export_pruned_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${MODEL_DIR}/model.ckpt-1 \
    --output_directory ${OUTPUT_DIR}

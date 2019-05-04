#!/bin/sh

PIPELINE_CONFIG_PATH=object_detection/samples/configs/prunable_ssd_mobilenet_v2_pets.config
MODEL_DIR=prunable_ssd_mobilenet_v2_pets

PYTHONPATH="./":"../lib_cnn":"$PYTHONPATH"
export PYTHONPATH

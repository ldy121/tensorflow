# From the tensorflow/models/research/ directory

#NUM_TRAIN_STEPS=50000

PIPELINE_CONFIG_PATH=object_detection/samples/configs/prunable_ssd_mobilenet_v2_pets.config
MODEL_DIR=prunable_ssd_mobilenet_v2_pets
NUM_TRAIN_STEPS=1
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

PYTHONPATH="./":"../lib_cnn":"$PYTHONPATH"
export PYTHONPATH

python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr

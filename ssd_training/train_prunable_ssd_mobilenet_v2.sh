# From the tensorflow/models/research/ directory
CONFIG_SCRIPT=./ssd_mobilenet_v2_config.sh
#NUM_TRAIN_STEPS=50000
NUM_TRAIN_STEPS=1
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

. ${CONFIG_SCRIPT}

python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr

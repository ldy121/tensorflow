from tensorflow.python.ops import random_ops

flower_class = 4;
imagenet_class = 1001;
num_class = imagenet_class;

batch_size, height, width = 1, 299, 299;
inputs = random_ops.random_uniform((batch_size, height, width, 3));

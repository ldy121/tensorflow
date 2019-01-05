from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

flower_class = 4;
imagenet_class = 1001;
num_class = imagenet_class;

batch_size, height, width, channel = None, 299, 299, 3;

def get_input() :
	ret = array_ops.placeholder(dtypes.float32,
		shape = (batch_size, height, width, channel), name = 'input');
	return ret;


import sys
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim.nets import inception_v3

import prunable_inception_v3
import inception_v3_parameter

num_class = inception_v3_parameter.imagenet_class;

def generate_prunable_inception_v3(sess) :
	with arg_scope(prunable_inception_v3.prunable_inception_v3_arg_scope) :
		input = inception_v3_parameter.get_input();
		final_endpoints, end_points = \
			prunable_inception_v3.prunable_inception_v3(input,
				num_classes = num_class, is_training = False);
		sess.run(tf.global_variables_initializer());

def generate_inception_v3(sess) :
	with arg_scope(inception_v3.inception_v3_arg_scope()) :
		input = inception_v3_parameter.get_input();
		final_endpoints, end_points = inception_v3.inception_v3(input,
				num_classes = num_class, is_training = False);
		sess.run(tf.global_variables_initializer());

def get_all_weights(sess) :
	ret = {};
	for i in tf.trainable_variables() :
		ret.update({i.name : sess.run(i)});
	return ret;

def print_all_weights(sess) :
	for i in tf.trainable_variables() :
		print i.name;
		print sess.run(i);

def reset_all_weights(sess) :
	for i in tf.trainable_variables() :
		tensor = sess.graph.get_tensor_by_name(i.name);
		zeros = np.zeros(i.shape);
		replace = i.assign(zeros);
		sess.run(replace);

def replace_all_weights(sess, weights) :
	for i in tf.trainable_variables() :
		if i.name in weights :
			tensor = sess.graph.get_tensor_by_name(i.name);
			replace = i.assign(weights[i.name]);
			sess.run(replace);

def save_graph(sess, file_path) :
	index = file_path.rfind('/');
	if index == -1 :
		file_path = './' + file_path;

	saver = tf.train.Saver();
	saver.save(sess, file_path);
	tf.train.write_graph(sess.graph_def, file_path[:index],
					file_path[index + 1:] + '.pbtxt');
	train_writer = tf.summary.FileWriter(file_path[:index]);
	train_writer.add_graph(sess.graph);

def restore_graph(sess, file_path) :
	saver = tf.train.Saver();
	saver.restore(sess, file_path);

if __name__ == '__main__' :
	if len(sys.argv) == 3 :
		graph = tf.Graph();
		with graph.as_default() :
			with tf.Session() as sess :
				generate_inception_v3(sess);
				restore_graph(sess, sys.argv[1]);
				weights = get_all_weights(sess);

		tf.reset_default_graph();

		graph = tf.Graph();
		with graph.as_default() :
			with tf.Session() as sess :
				generate_prunable_inception_v3(sess);
				replace_all_weights(sess, weights);
				save_graph(sess, sys.argv[2]);
	else :
		print (sys.argv[0] + ' [ pretrained checkpoint ] \
					[ checkpoint path to be stored ]');


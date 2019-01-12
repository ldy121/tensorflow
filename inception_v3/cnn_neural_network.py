import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim import nets
import prunable_inception_v3 as inception_v3_prun
import inception_v3_parameter

class cnn_neural_network(object) :
	def __init__(self, name, sess) :
		self.name = name;
		self.sess = sess;

	def node_iteration(self, func) :
		for i in tf.trainable_variables() :
			func(i);

	def replace_node(self, tensor, values) :
		replace = tensor.assign(values);
		self.sess.run(replace);

	def get_all_weights(self) :
		ret = {};
		def _get_all_weights(i) :
			ret.update({i.name : self.sess.run(i)});
		self.node_iteration(_get_all_weights);
		return ret;

	def reset_all_weights(self) :
		def _reset_all_weights(i) :
			zeros = np.zeros(i.shape);
			self.replace_node(i, zeros);
		self.node_iteration(_reset_all_weights);

	def replace_graph(self, weight_map) :
		def _replace_graph(i) :
			if i.name in weight_map :
				self.replace_node(i, weight_map[i.name]);
		self.node_iteration(_replace_graph);

	def print_all_weights(self) :
		all_weights = self.get_all_weights();
		for i in all_weights.keys() :
			print (i);
			print (all_weights[i]);

	def save_graph(self, file_path) :
		index = file_path.rfind('/');
		if index == -1 :
			file_path = './' + file_path;

		saver = tf.train.Saver();
		saver.save(self.sess, file_path);
		tf.train.write_graph(self.sess.graph_def, file_path[:index],
					file_path[index + 1:] + '.pbtxt');
		train_writer = tf.summary.FileWriter(file_path[:index]);
		train_writer.add_graph(self.sess.graph);

	def restore_graph(self, file_path) :
		saver = tf.train.Saver();
		saver.restore(self.sess, file_path);

class inception_v3(cnn_neural_network) :
	def __init__(self, sess) :
		with arg_scope(nets.inception_v3.inception_v3_arg_scope()) :
			input = inception_v3_parameter.get_input();
			final_endpoints, end_points = \
				nets.inception_v3.inception_v3(input, \
				num_classes = inception_v3_parameter.num_class, \
				is_training = False);
			sess.run(tf.global_variables_initializer());
		cnn_neural_network.__init__(self, self.__class__.__name__, sess);

class prunable_inception_v3(cnn_neural_network) :
	def __init__(self, sess) :
		with arg_scope(inception_v3_prun.prunable_inception_v3_arg_scope) :
			input = inception_v3_parameter.get_input();
			final_endpoints, end_points = \
				inception_v3_prun.prunable_inception_v3(input, \
				num_classes = inception_v3_parameter.num_class, \
				is_training = False);
			sess.run(tf.global_variables_initializer());
		cnn_neural_network.__init__(self, self.__class__.__name__, sess);

import sys
import tensorflow as tf
from neural_network import inception_v3
from neural_network import prunable_inception_v3

def compare_inception_v3(src1, src2) :
	graph = tf.Graph();
	with graph.as_default() :
		with tf.Session() as sess :
			network = inception_v3(sess);
			network.restore_graph(src1);
			weights = network.get_all_weights();
			network.restore_graph(src2);
			total_weight, equal_weight = network.comp_graph(weights);
			print ('Total weight : %d / Equal weight : %d' % \
					(total_weight, equal_weight));

def generate_pretrained_prunable_inception_v3(src, dst) :
	graph = tf.Graph();
	with graph.as_default() :
		with tf.Session() as sess :
			network = inception_v3(sess);
			network.restore_graph(src);
			weights = network.get_all_weights();

	tf.reset_default_graph();

	graph = tf.Graph();
	with graph.as_default() :
		with tf.Session() as sess :
			network = prunable_inception_v3(sess);
			network.replace_graph(weights);
			network.save_graph(dst);

def generate_zero_inception_v3(dst) :
	with tf.Session() as sess :
		network = inception_v3(sess);
		network.reset_all_weights();
		network.save_graph(dst);

def generate_pretrained_inception_v3(src, dst) :
	with tf.Session() as sess :
		network = inception_v3(sess);
		network.restore_graph(src);
		network.save_graph(dst);

def generate_prunable_inception_v3(dst) :
	with tf.Session() as sess :
		network = prunable_inception_v3(sess);
		network.reset_all_weights();
		network.save_graph(dst);

def print_inception_v3(src) :
	with tf.Session() as sess :
		network = inception_v3(sess);
		network.restore_graph(src);
		network.print_all_weights();

def print_prunable_inception_v3(src) :
	with tf.Session() as sess :
		network = prunable_inception_v3(sess);
		network.restore_graph(src);
		network.print_all_weights();

def get_func(op, num_arg) :
	min_arg = min(len(func), num_arg);
	for i in range(min_arg + 1) :
		func_list = func[i];
		if op in func_list :
			if i == num_arg :
				return func_list[op];
			break;
	return None;

def print_help(cmd) :
	for i in func :
		for j in i.keys() :
			print (cmd + ' ' + help[j]);

help = {
	'pretrained_prunable_inception_v3' : 'pretrained_prunable_inception_v3 [ src checkpoint ] [ dst checkpoint ]',
	'pretrained_inception_v3' : 'pretrained_inception_v3 [ src checkpoint ] [ dst checkpoint ]',
	'zero_inception_v3' : 'zero_inception_v3 [dst checkpoint]',
	'prunable_inception_v3' : 'prunable_inception_v3 [dst checkpoint]',
	'print_inception_v3' : 'print_inception_v3 [src checkpoint]',
	'print_prunable_inception_v3' : 'print_prunable_inception_v3 [src checkpoint]',
	'compare_inception_v3' : 'compare_inception_v3 [src1 checkpoint] [src2 checkpoint]',
};

func_arg2 = {
	'pretrained_prunable_inception_v3' : generate_pretrained_prunable_inception_v3,
	'pretrained_inception_v3' : generate_pretrained_inception_v3,
	'compare_inception_v3' : compare_inception_v3
};

func_arg1 = {
	'zero_inception_v3' : generate_zero_inception_v3,
	'prunable_inception_v3' : generate_prunable_inception_v3,
	'print_inception_v3' : print_inception_v3,
	'print_prunable_inception_v3' : print_prunable_inception_v3
};

func_arg0 = {
};

func = [func_arg0, func_arg1, func_arg2];

if __name__ == '__main__' :
	f = None;
	if len(sys.argv) >= 3 :
		f = get_func(sys.argv[1], len(sys.argv) - 2);
	if f == None :
		print_help(sys.argv[0]);
	else :
		f (*(sys.argv[2:]));


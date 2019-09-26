import tensorflow.compat.v1 as tf
from src.fruit import Fruit

class Environment:

	def __init__(self, scope="global_env", starting_index=0,
				load_path="dataset/dataset/", defects_thresholds=[160]):

		self.scope = scope
		self.defects_thresholds = defects_thresholds

		with tf.variable_scope(scope):
			self.index = tf.Variable(starting_index-1, dtype=tf.int32,
									name='index', trainable=False)
			self.load_path = tf.Variable(load_path, dtype=tf.string,
									name='load_path', trainable=False)

	@staticmethod
	def sync(sess, from_scope, to_scope):

		from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
		to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)

		synchronizer = []
		for from_var, to_var in zip(from_vars, to_vars):
			synchronizer.append(to_var.assign(from_var))

		sess.run(synchronizer)

	def load_fruit(self, sess):

		Environment.sync(sess, "global_env", self.scope)
		load_path = sess.run(self.load_path).decode("utf-8")

		temp_index = sess.run(self.index)+1
		fruit = Fruit(temp_index, load_path, self.defects_thresholds)

		while not fruit.defects_tot:
			temp_index += 1
			fruit = Fruit(temp_index, load_path, self.defects_thresholds)

		self.fruit = fruit
		sess.run(self.index.assign(temp_index))
		
		Environment.sync(sess, self.scope, "global_env")
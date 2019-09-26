import tensorflow.compat.v1 as tf
from src.fruit import Fruit

class Environment:

	def __init__(self, scope, starting_index=0,
				load_path="dataset/dataset/", defects_thresholds=[160]):

		self.scope = scope
		self.load_path = load_path
		self.defects_thresholds = defects_thresholds

		with tf.variable_scope(scope):
			self.index = tf.Variable(starting_index-1, dtype=tf.int32,
									name='index', trainable=False)

	@staticmethod
	def sync(sess, from_scope, to_scope):

		from_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
		to_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)

		sess.run(to_var[0].assign(from_var[0]))

	def load_fruit(self, sess):

		Environment.sync(sess, "global_env", self.scope)

		temp_index = sess.run(self.index)+1
		fruit = Fruit(temp_index, self.load_path, self.defects_thresholds)

		while not fruit.defects_tot:
			temp_index += 1
			fruit = Fruit(temp_index, self.load_path, self.defects_thresholds)

		self.fruit = fruit
		sess.run(self.index.assign(temp_index))
		
		Environment.sync(sess, self.scope, "global_env")

import tensorflow.compat.v1 as tf
from src.fruit import Fruit

class Environment:

	def __init__(self, scope, starting_index=0,
				load_path="dataset/dataset/", defects_thresholds=[160]):

		self.load_path = load_path
		self.defects_thresholds = defects_thresholds

		with tf.variable_scope(scope):
			self.index = tf.Variable(starting_index-1, dtype=tf.int32,
									name='index', trainable=False)

	def load_fruit(self, sess):

		index = sess.run(self.index)+1
		fruit = Fruit(index, self.load_path, self.defects_thresholds)

		while not fruit.defects_tot:
			index += 1
			fruit = Fruit(index, self.load_path, self.defects_thresholds)

		self.fruit = fruit
		sess.run(self.index.assign(index))
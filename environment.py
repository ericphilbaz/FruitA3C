import tensorflow as tf
from src.fruit import Fruit
import numpy as np

class Environment:
	"""
	Used to manage Fruit objects and interaction with agents
	"""

	def __init__(self, scope="global_env", starting_index=0, final_index=100,
				load_path="dataset/dataset/", defects_thresholds=[160]):
		"""
		Creates a new environment that automatically loads new fruits

		Parameters
		----------
		scope : str
			scope of the environment
		starting_index : int
			starting index of the first fruit
		load_path : str
			load path of the fruits
		defects_thresholds : list
			list of thresholds for labeling
		"""

		self.scope = scope

		with tf.variable_scope(scope):
			self.index = tf.Variable(starting_index-1, dtype=tf.int64,
									name='index', trainable=False)
			self.final_index = tf.Variable(final_index, dtype=tf.int64,
									name='final_index', trainable=False)
			self.load_path = tf.Variable(load_path, dtype=tf.string,
									name='load_path', trainable=False)
			self.defects_thresholds = tf.Variable(defects_thresholds, dtype=tf.int64,
									name='defects_thresholds', trainable=False)

	@staticmethod
	def sync(sess, from_scope, to_scope):
		"""
		Syncs the variables between two different environments
		"""

		from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
		to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)

		synchronizer = []
		for from_var, to_var in zip(from_vars, to_vars):
			synchronizer.append(to_var.assign(from_var))

		sess.run(synchronizer)

	def load_fruit(self, sess, coord):
		"""
		Loads a new Fruit with defects

		Parameters
		----------
		sess : tf.Session()
			Tensorflow session used to run sync parameters
		"""

		Environment.sync(sess, "global_env", self.scope)
		self.fruit = None

		index = sess.run(self.index)+1
		final_index = sess.run(self.final_index)
		load_path = sess.run(self.load_path).decode("utf-8")
		defects_thresholds = sess.run(self.defects_thresholds)

		if index < final_index:
			try:
				fruit = Fruit(index, load_path, defects_thresholds)
				while not fruit.defects_tot:
					index += 1
					fruit = Fruit(index, load_path, defects_thresholds)

				sess.run(self.index.assign(index))
				Environment.sync(sess, self.scope, "global_env")

				self.fruit = fruit				
			except:
				coord.request_stop()
		else:
			coord.request_stop()

	def get_state(self):
		"""
		Gets the state vector

		Returns
		-------
		state : array
			state vector
		"""

		shots_progress = self.fruit.shots_index/self.fruit.shots_tot
		defects_progress = self.fruit.defects_index/self.fruit.defects_tot
		uuid_progress = self.fruit.defects_identified/self.fruit.defects_index if self.fruit.defects_index else 0.0

		return np.array([shots_progress, defects_progress, uuid_progress]).reshape((1, 3))
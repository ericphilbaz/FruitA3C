import tensorflow as tf
from src.fruit import Fruit
import numpy as np
from uuid import uuid4

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
		self.load_path = tf.constant(load_path, dtype=tf.string,
									name='load_path')

		with tf.variable_scope(scope):
			self.index = tf.Variable(starting_index, dtype=tf.int64,
									name='index', trainable=False)
			self.final_index = tf.Variable(final_index, dtype=tf.int64,
									name='final_index', trainable=False)
			self.defects_thresholds = tf.Variable(defects_thresholds, dtype=tf.int64,
									name='defects_thresholds', trainable=False)

	@staticmethod
	def sync(sess, to_scope, from_scope="global_env"):
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

		Environment.sync(sess, self.scope)
		self.fruit = None

		index = sess.run(self.index)
		final_index = sess.run(self.final_index)
		load_path = sess.run(self.load_path).decode("utf-8")
		defects_thresholds = sess.run(self.defects_thresholds)

		if index < final_index:
			try:
				fruit = Fruit(index, load_path, defects_thresholds)
				while not fruit.defects_tot:
					index += 1
					fruit = Fruit(index, load_path, defects_thresholds)

				sess.run(self.index.assign(index+1))
				Environment.sync(sess, "global_env", self.scope)

				self.fruit = fruit

				self.indices_analyzed = set()

				# self.answers_dict = {key: set() for key in self.fruit.defects_indices}
				# self.uuids_dict = dict()
				# self.loss = self.get_loss()
				
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

		shots_progress = self.fruit.shot_index/self.fruit.shots_tot
		defects_progress = self.fruit.defect_index/self.fruit.defects_tot
		uuid_progress = self.fruit.defects_identified/self.fruit.defect_index if self.fruit.defect_index else 0.0

		return np.array([shots_progress, defects_progress, uuid_progress]).reshape((1, 3))

	def add_uuid(self, defect, defect_matched=None):
		"""
		Adds uuid to the defect analyzed

		Parameters
		----------
		defect : Defect
			defect to apply action
		defect_matched : Defect
			defect matched

		Returns
		-------
		identified : int
			number of defects identified in the process
		"""

		uuid = uuid4()
		identified = 0

		if not defect_matched:
			defect.uuid = uuid
			identified += 1
		else:
			if not defect_matched.uuid:
				defect_matched.uuid = uuid
				identified += 1
			defect.uuid = defect_matched.uuid
			identified += 1

		return identified

	def apply_action(self, action, defect, defect_matched):
		"""
		Apply action to the defect currently analyzed

		Parameters
		----------
		defect : Defect
			defect to apply action
		action : str
			action to apply
		defect_matched : Defect
			defect matched

		Returns
		-------
		reward : int
			reward for the applied action
		"""

		if action is "wait":
			reward = 0
			identified = 0
		elif action is "match":
			if defect.index == defect_matched.index:
				reward = +1
			else:
				reward = -1
			self.indices_analyzed.add(defect.index)
			identified = self.add_uuid(defect, defect_matched)
		elif action is "new":
			if defect.index == defect_matched.index:
				reward = -1
			else:
				if defect.index in self.indices_analyzed:
					reward = -1
				else:
					reward = +1
			self.indices_analyzed.add(defect.index)
			identified = self.add_uuid(defect)

		self.fruit.defects_identified += identified

		return reward
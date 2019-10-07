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
				Environment.sync(sess, "global_env", self.scope)

				self.fruit = fruit
				self.answers_dict = {key: set() for key in self.fruit.defects_indices}
				self.uuids_dict = dict()
				self.new_loss = -sum([abs(len(l)-1) for key, l in self.answers_dict.items()])
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

	# def add_uuid(self, defect, defect_matched=None):

		# new_uuid = uuid4()

		# if not defect_matched:
		# 	defect.ID = new_uuid
		# 	self.answers_dict[defect.index].append(new_uuid)
		# 	self.fruit.defects_identified += 1
		# 	self.uuids_dict.append(new_uuid)
		# else:
		# 	if not defect_matched.ID:
		# 		defect.ID = new_uuid
		# 		defect_matched.ID = new_uuid
		# 		self.answers_dict[defect.index].append(new_uuid)
		# 		if defect_matched.index != defect.index:
		# 			self.answers_dict[defect_matched.index].append(new_uuid)
		# 		self.fruit.defects_identified += 2
		# 		self.uuids_dict.append(new_uuid)
		# 	else:
		# 		defect.ID = defect_matched.ID
		# 		if defect_matched.index != defect.index:
		# 			self.answers_dict[defect.index].append(defect_matched.ID)
		# 		self.fruit.defects_identified += 1

	# def get_reward(self):

	# 	new_loss = -sum([abs(len(l)-1) for key, l in self.answers_dict.items()])

	# 	reward = new_loss-self.new_loss
	# 	self.new_loss = new_loss

	# 	return reward

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
		"""
		reward = 0
		uuid = uuid4()

		if action is "new":
			self.answers_dict[defect.index].add(uuid)
			self.uuids_dict[uuid] = set([defect.index])
			defect.uuid = uuid
			self.fruit.defects_identified += 1
		elif action is "match":
			if defect_matched.uuid:
				self.answers_dict[defect.index].add(defect_matched.uuid)
				self.uuids_dict[defect_matched.uuid].add(defect.index)
				defect.uuid = defect_matched.uuid
				self.fruit.defects_identified += 1
			else:
				self.answers_dict[defect_matched.index].add(uuid)
				self.uuids_dict[uuid] = set([defect_matched.index])
				defect_matched.uuid = uuid
				self.answers_dict[defect.index].add(defect_matched.uuid)
				self.uuids_dict[defect_matched.uuid].add(defect.index)
				defect.uuid = defect_matched.uuid
				self.fruit.defects_identified += 2 if defect != defect_matched else 1
		else:
			pass

		# 	self.add_uuid(defect)
		# elif action is "match":
		# 	self.add_uuid(defect, defect_matched)
		# else:
		# 	pass

		# reward = self.get_reward()
		return reward
from src.environment import Environment
from A3C_network import A3C_Network
import numpy as np

class Agent:
	"""
	Used to implement the A3C algorithm
	"""

	def __init__(self, index,
				n_inputs_policy, n_inputs_matching, n_actions_policy):
		"""
		Initializes a new agent

		Parameters
		----------
		index : int
			used to set up the agent name
		"""

		self.name = "agent_{0}".format(index)
		self.local_env = Environment("env_{0}".format(index))
		self.local_net = A3C_Network("net_{0}".format(index), n_inputs_policy,
									n_inputs_matching, n_actions_policy)

	def find_match(self, sess, defect):

		defects_to_match = self.local_env.fruit.defects[:self.local_env.fruit.shots_index]
		defect_matched, val_matched = None, -1

		for s in defects_to_match:
			for d in s:
				val = sess.run(self.local_net.match,
								feed_dict={self.local_net.matching_vector: defect - d})
				if val >= val_matched:
					defect_matched = d

		return defect_matched

	def train(self, sess, coord, lock):
		"""
		Trains the agent

		Parameters
		----------
		sess : tf.Session()
			TensorFlow session used to run the train
		coord : tf.Coordinator()
			TensorFlow coordinator used to coordinate fruit loading
		lock : threading.Lock()
			used to properly load a new fruit by correctly syncing the local environment
		"""

		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():

				self.local_net.sync(sess)
				lock.acquire()
				self.local_env.load_fruit(sess, coord)
				lock.release()
				
				if self.local_env.fruit is not None:

					for defect in self.local_env.fruit:
						defect_matched = self.find_match(sess, defect)
						print(defect_matched)
						state = self.local_env.get_state()
						print("analyzing...", defect.shot_name)
						reward = self.local_env.apply_action(defect, "new")
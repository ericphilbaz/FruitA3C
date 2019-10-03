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

		self.actions_available = ["new", "match"]

	def find_match(self, sess, defect):
		"""
		Find the most matching defect based on the matching result

		Parameters
		----------
		sess : tf.Session()
			TensorFlow session used to run the function
		defect : Defect
			defect to match

		Returns
		-------
		defect_matched : Defect
			defect matched
		"""

		defects_to_match = self.local_env.fruit.defects[:self.local_env.fruit.shot_index]
		defect_matched, val_matched = defect, -1

		for s in defects_to_match:
			for d in s:
				val = sess.run(self.local_net.match,
								feed_dict={self.local_net.matching_vector: defect - d})
				if val >= val_matched:
					defect_matched = d

		return defect_matched

	def policy(self, sess, state, defect, defect_matched):
		"""
		Evaluates the policy

		Parameters
		----------
		sess : tf.Session()
			TensorFlow session used to run the function
		state : array
			state vector
		defect : Defect
			defect to evaluate
		defect_matched : Defect
			defect matched

		Returns
		-------
		action : str
			action to perform from self.actions_available
		action_idx : int
			action to perform as index
		"""
		
		actions_distribution = sess.run(self.local_net.policy,
										feed_dict={self.local_net.input_vector:state,
										self.local_net.matching_vector:defect-defect_matched})
		action_idx = np.random.choice(len(actions_distribution[0]), p=actions_distribution[0])

		return self.actions_available[action_idx], action_idx

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
						print(defect_matched.shot_name if defect_matched else defect_matched)

						state = self.local_env.get_state()
						# print(state)

						print("analyzing...", defect.shot_name)

						action, action_idx = self.policy(sess, state, defect, defect_matched)
						# print(action, action_idx)

						reward = self.local_env.apply_action(action, defect, defect_matched)
						# print(reward)
						print()
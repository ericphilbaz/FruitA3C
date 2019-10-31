from src.environment import Environment
from src.A3C_network import A3C_Network
from src.utils import discount
import numpy as np
import tensorflow as tf

class Agent:
	"""
	Used to implement the A3C algorithm
	"""

	def __init__(self, n_agent, n_inputs_policy,
				n_inputs_matching, n_actions_policy, trainer,
				load_path="dataset/dataset/", model_path="./model"):
		"""
		Initializes a new agent

		Parameters
		----------
		n_agent : int
			used to set up the agent name
		"""

		self.name = "agent_{0}".format(n_agent)
		self.model_path = model_path

		self.local_env = Environment("env_{0}".format(n_agent), load_path=load_path)
		self.local_net = A3C_Network("net_{0}".format(n_agent), n_inputs_policy,
									n_inputs_matching, n_actions_policy, trainer)

		# self.actions_available = ["new", "match", "wait"]
		self.actions_available = ["new", "match"]

		with tf.variable_scope(self.name):
			self.episodes = tf.Variable(0, dtype=tf.int64,
										name='episodes', trainable=False)

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

	def value(self, sess, state, defect, defect_matched):
		"""
		Evaluates the value

		Parameters
		----------
		sess : tf.Session()
			TensorFlow session used to run the function

		"""

		value = sess.run(self.local_net.value,
						feed_dict={self.local_net.input_vector:state,
									self.local_net.matching_vector:defect-defect_matched})

		return value

	def update(self, sess, analysis, gamma):
		"""
		Updates the global network by applying gradients

		Parameters
		----------
		sess : tf.Session()
			TensorFlow session used to run the function
		analysis : array
			array of parameters used to run the function
		gamma : float
			discount parameter for reinforcement learning

		Returns
		-------
		value_loss/length, policy_loss/length, entropy/length, loss/length : float
			parameters used to evaluate the model
		"""
		
		analysis = np.array(analysis)

		states = analysis[:, 0]
		matching = analysis[:, 1]
		actions = analysis[:, 2]
		rewards = analysis[:, 3]
		values = analysis[:, 4]

		rewards_plus = np.asarray(rewards.tolist() + [0.0])
		discounted_rewards = discount(rewards_plus, gamma)[:-1]

		values_plus = np.asarray(values.tolist() + [0.0])
		advantages = rewards + gamma * values_plus[1:] - values_plus[:-1]

		feed_dict = {self.local_net.target_value:discounted_rewards,
					self.local_net.input_vector:np.vstack(states),
					self.local_net.matching_vector:np.vstack(matching),
					self.local_net.actions:actions,
					self.local_net.advantages:advantages}
		value_loss, policy_loss, entropy, loss, _ = sess.run([self.local_net.value_loss,
															self.local_net.policy_loss,
															self.local_net.entropy,
															self.local_net.loss,
															self.local_net.apply_grads],
															feed_dict=feed_dict)

		length = len(analysis)
		return value_loss/length, policy_loss/length, entropy/length, loss/length

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

		self.summary_writer = tf.summary.FileWriter("./graphs/train_"+str(self.name))
		episodes = sess.run(self.episodes)

		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():

				self.local_net.sync(sess)
				lock.acquire()
				self.local_env.load_fruit(sess, coord)
				lock.release()
				
				if self.local_env.fruit is not None:

					fruit_analysis = []
					fruit_values = []
					fruit_total_reward = 0

					for defect in self.local_env.fruit:

						defect_matched = self.find_match(sess, defect)
						state = self.local_env.get_state()

						value = self.value(sess, state, defect, defect_matched)
						action, action_idx = self.policy(sess, state, defect, defect_matched)
						reward = self.local_env.apply_action(action, defect, defect_matched)

						fruit_analysis.append([state, defect-defect_matched, action_idx,
																				reward, value])
						fruit_values.append(value)
						fruit_total_reward += reward

					fruit_avg_reward = fruit_total_reward/self.local_env.fruit.defects_tot
					fruit_avg_values = np.mean(fruit_values)

					v_l, p_l, e_l, t_l = self.update(sess, fruit_analysis, 1)

					if episodes % 8 == 0 and episodes != 0:

						summary = tf.Summary()
						summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
						summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
						summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
						summary.value.add(tag='Losses/Total Loss', simple_value=float(t_l))
						summary.value.add(tag='Performances/Fruit Average Reward', simple_value=float(fruit_avg_reward))
						summary.value.add(tag='Performances/Fruit Average Value', simple_value=float(fruit_avg_values))

						self.summary_writer.add_summary(summary, episodes)
						self.summary_writer.flush()

					episodes += 1
			sess.run(self.episodes.assign(episodes))

	def test(self, sess, coord):

		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():

				self.local_net.sync(sess)
				self.local_env.load_fruit(sess, coord)
				
				if self.local_env.fruit is not None:

					print("Running test on fruit number", self.local_env.fruit.index)

					fruit_analysis = []
					fruit_values = []
					fruit_total_reward = 0

					for defect in self.local_env.fruit:

						defect_matched = self.find_match(sess, defect)
						state = self.local_env.get_state()

						value = self.value(sess, state, defect, defect_matched)
						action, action_idx = self.policy(sess, state, defect, defect_matched)
						reward = self.local_env.apply_action(action, defect, defect_matched)

						fruit_analysis.append([state, defect-defect_matched, action_idx,
																				reward, value])
						fruit_values.append(value)
						fruit_total_reward += reward

						print("############################################################")
						print("Defect", defect.index,
							"of shot", defect.shot_name,
							"matched with defect", defect_matched.index,
							"of shot", defect_matched.shot_name)
						print("Action applied is", "\""+action+"\"",
							"with reward", reward)

					fruit_avg_reward = fruit_total_reward/self.local_env.fruit.defects_tot
					fruit_avg_values = np.mean(fruit_values)

					print()
					print("Fruit average reward is:", fruit_avg_reward)
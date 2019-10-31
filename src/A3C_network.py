import tensorflow as tf
from src.utils import normalized_columns_initializer

class A3C_Network:

	def __init__(self, scope="global_net", n_inputs_policy=3,
				n_inputs_matching=3, n_actions_policy=2, trainer=None):
		"""
		Setting up the network

		Parameters
		----------
		scope : str
			variable scope
		n_inputs_policy : int
			number of policy inputs
		n_inputs_matching : int
			number of matching inputs
		n_actions_policy : int
			number of policy outputs
		trainer : tf.trainer
			trainer of the network
		"""

		self.scope = scope
		self.n_inputs_policy = n_inputs_policy
		self.n_inputs_matching = n_inputs_matching
		self.n_actions_policy = n_actions_policy

		with tf.variable_scope(scope):

			# matching part
			self.matching_vector = tf.placeholder(name="matching_vector",
												shape=[None, n_inputs_matching], dtype=tf.float32)

			self.W_1m = tf.get_variable(name="W_1m", shape=[n_inputs_matching, 4],
											initializer=tf.initializers.glorot_uniform())
			self.layer_1m = tf.nn.relu(tf.matmul(self.matching_vector, self.W_1m,
												name="matmul_1m"), name="layer_1m")

			self.W_match = tf.get_variable(name="W_match", shape=[4, 1],
											initializer=normalized_columns_initializer(0.01))
			self.match = tf.math.sigmoid(tf.matmul(self.layer_1m, self.W_match,
													name="matmul_match"), name="match")

			# policy part
			self.input_vector = tf.placeholder(name="input_vector",
												shape=[None, n_inputs_policy], dtype=tf.float32)
			self.input_vector_extended = tf.concat([self.input_vector, self.match], 1)

			self.W_1p = tf.get_variable(name="W_1p", shape=[n_inputs_policy+1, 4],
											initializer=tf.initializers.glorot_uniform())
			self.layer_1p = tf.nn.relu(tf.matmul(self.input_vector_extended, self.W_1p,
												name="matmul_1p"), name="layer_1p")

			self.W_policy = tf.get_variable(name="W_policy", shape=[4, n_actions_policy],
									initializer=normalized_columns_initializer(0.01))
			self.policy = tf.nn.softmax(tf.matmul(self.layer_1p, self.W_policy,
												name="matmul_policy"), name="policy")

			self.W_value = tf.get_variable(name="W_value", shape=[4, 1],
									initializer=normalized_columns_initializer(1.0))
			self.value = tf.matmul(self.layer_1p, self.W_value, name="value")

			if scope != "global_net":

				# training part
				self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot = tf.one_hot(self.actions, n_actions_policy, dtype=tf.float32)
				self.target_value = tf.placeholder(shape=[None], dtype=tf.float32)
				self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

				self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

				self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_value - tf.reshape(self.value,[-1])))
				self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
				self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
				self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss,local_vars)

				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(zip(self.gradients,global_vars))


	def sync(self, sess):
		"""
		Syncs the variables between two different environments

		Parameters
		----------
		sess : tf.Session()
			TensorFlow session used to run the train
		"""

		from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global_net")
		to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

		synchronizer = []
		for from_var, to_var in zip(from_vars, to_vars):
			synchronizer.append(to_var.assign(from_var))

		sess.run(synchronizer)
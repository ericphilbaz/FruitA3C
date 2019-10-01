import tensorflow as tf
from utils import normalized_columns_initializer

class A3C_Network:

	def __init__(self, scope="global_net", n_inputs_policy=3,
				n_inputs_matching=3, n_policy_actions=2):

		self.scope = scope

		with tf.variable_scope(scope):

			# matching part
			self.matching_vector = tf.placeholder(name="matching_vector",
												shape=[1, n_inputs_matching], dtype=tf.float32)

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
												shape=[1, n_inputs_policy], dtype=tf.float32)
			self.input_vector_extended = tf.concat([self.input_vector, self.match], 1)

			self.W_1p = tf.get_variable(name="W_1p", shape=[n_inputs_policy+1, 4],
											initializer=tf.initializers.glorot_uniform())
			self.layer_1p = tf.nn.relu(tf.matmul(self.input_vector_extended, self.W_1p,
												name="matmul_1p"), name="layer_1p")

			self.W_policy = tf.get_variable(name="W_policy", shape=[4, n_policy_actions],
									initializer=normalized_columns_initializer(0.01))
			self.policy = tf.nn.softmax(tf.matmul(self.layer_1p, self.W_policy,
												name="matmul_policy"), name="policy")

			self.W_value = tf.get_variable(name="W_value", shape=[4, 1],
									initializer=normalized_columns_initializer(1.0))
			self.value = tf.matmul(self.layer_1p, self.W_value, name="value")
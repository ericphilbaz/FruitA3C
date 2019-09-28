import tensorflow as tf
from utils import normalized_columns_initializer

class A3C_Network:

	def __init__(self, scope="global_env", n_inputs_policy=3,
				n_inputs_matching=3, n_policy_actions=2):

		self.scope = scope

		with tf.variable_scope(scope):

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
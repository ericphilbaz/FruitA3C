import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'off'
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from environment import Environment

global_env = Environment("global_env")

local_env_0 = Environment("env_0")
local_env_1 = Environment("env_1")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	local_env_0.load_fruit(sess)
	print(local_env_0.fruit.index)

	local_env_1.load_fruit(sess)
	print(local_env_1.fruit.index)
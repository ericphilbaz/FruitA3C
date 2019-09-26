import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'off'
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from environment import Environment
from agent import Agent
import multiprocessing, threading

global_env = Environment("global_env", load_path="dataset/dataset2/")

with tf.Session() as sess:

	# n_agents = 1
	n_agents = multiprocessing.cpu_count()

	lock = threading.Lock()

	agents = []
	for i in range(n_agents):
		agents.append(Agent(i))

	sess.run(tf.global_variables_initializer())

	agents_threads = []
	for agent in agents:
		agent_train = lambda: agent.train(sess, lock)
		t = threading.Thread(target=(agent_train))
		t.start()
		agents_threads.append(t)

	for t in agents_threads:
		t.join()
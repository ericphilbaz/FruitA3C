import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from src.environment import Environment
from A3C_network import A3C_Network
from agent import Agent
import multiprocessing, threading

with tf.device('/gpu:0'):
	trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

	global_env = Environment(final_index=1)
	global_net = A3C_Network()

	n_agents = 1
	# n_agents = multiprocessing.cpu_count()

	agents = []
	for i in range(n_agents):
		agents.append(Agent(i, global_net.n_inputs_policy, global_net.n_inputs_matching,
							global_net.n_actions_policy, trainer))

	coord = tf.train.Coordinator()
	lock = threading.Lock()


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	sess.run(tf.global_variables_initializer())

	agents_threads = []
	for agent in agents:
		agent_train = lambda: agent.train(sess, coord, lock)
		t = threading.Thread(target=(agent_train))
		t.start()
		agents_threads.append(t)

	for t in agents_threads:
		t.join()

	writer = tf.summary.FileWriter('./graphs', sess.graph)
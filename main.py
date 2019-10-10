import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from src.environment import Environment
from src.A3C_network import A3C_Network
from src.agent import Agent
import multiprocessing, threading

load_path = "dataset/dataset/"
model_path = './model'
load_model = False

with tf.device('/gpu:0'):
	global_episodes = tf.Variable(0, dtype=tf.int32,name='global_episodes',trainable=False)
	trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

	global_env = Environment(load_path=load_path, final_index=200)
	global_net = A3C_Network()

	# n_agents = 2
	n_agents = multiprocessing.cpu_count()

	agents = []
	for i in range(n_agents):
		agents.append(Agent(i, global_net.n_inputs_policy, global_net.n_inputs_matching,
							global_net.n_actions_policy, trainer, global_episodes,
							load_path, model_path))

	lock = threading.Lock()
	saver = tf.train.Saver(max_to_keep=5)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	
	coord = tf.train.Coordinator()

	if load_model == True:
		print ('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())

	agents_threads = []
	for agent in agents:
		agent_train = lambda: agent.train(sess, coord, lock, saver)
		t = threading.Thread(target=(agent_train))
		t.start()
		agents_threads.append(t)

	for t in agents_threads:
		t.join()

	writer = tf.summary.FileWriter('./graphs', sess.graph)
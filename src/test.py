import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from src.environment import Environment
from src.A3C_network import A3C_Network
from src.agent import Agent
import multiprocessing, threading

def run(load_path, model_path, index, load_model):

	with tf.device('/gpu:0'):
		trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

		global_env = Environment(load_path=load_path, starting_index=index, final_index=index+1)
		global_net = A3C_Network()
		agent = Agent(0, global_net.n_inputs_policy, global_net.n_inputs_matching,
						global_net.n_actions_policy, trainer, load_path, model_path)

		saver = tf.train.Saver()

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		
		coord = tf.train.Coordinator()

		if load_model:
			ckpt = tf.train.get_checkpoint_state(model_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			sess.run(global_env.index.assign(index))
			sess.run(global_env.final_index.assign(index+1))
		else:
			sess.run(tf.global_variables_initializer())

		agent_test = lambda: agent.test(sess, coord)
		t = threading.Thread(target=(agent_test))
		t.start()
		t.join()
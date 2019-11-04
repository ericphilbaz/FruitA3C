import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from src.model.environment import Environment
from src.model.A3C_network import A3C_Network
from src.model.agent import Agent
import multiprocessing, threading

from math import ceil
import time, datetime

def run_batch(n_agents, load_path, model_path, starting_index, final_index, load_model):

	with tf.device('/gpu:0'):

		trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

		global_env = Environment(load_path=load_path,
								starting_index=starting_index, final_index=final_index)
		global_net = A3C_Network()

		agents = []
		for i in range(n_agents):
			agents.append(Agent(i, global_net.n_inputs_policy, global_net.n_inputs_matching,
								global_net.n_actions_policy, trainer, load_path, model_path))

		lock = threading.Lock()
		saver = tf.train.Saver(max_to_keep=1)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		
		coord = tf.train.Coordinator()

		if load_model == True:
			ckpt = tf.train.get_checkpoint_state(model_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			sess.run(global_env.final_index.assign(final_index))
		else:
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
		saver.save(sess, model_path + "/model" + str(final_index) + ".cptk")

def run(n_agents, load_path, model_path, starting_index, final_index, batch, load_model):

	load_model = load_model

	total_batches = ceil(final_index/batch)
	times = []
	remaining_time = "unknown"

	for i in range(starting_index, final_index, batch):
		actual_batch = int(i/batch + 1)
		remaining_batches = total_batches - actual_batch

		print("Batch", actual_batch,
			"over", total_batches,
			"total batches, estimated time left:", str(remaining_time).split(".")[0],
			end="\r", flush=True)

		start = time.time()

		p = multiprocessing.Process(target=(run_batch), args=(n_agents, load_path,
														model_path, i,
														i+batch, load_model))
		p.start()
		p.join()
	
		times.append(time.time() - start)

		if len(times) > 20:	
			del times[0]

		if remaining_batches:
			mean_time = sum(times)/len(times)
			remaining_time = datetime.timedelta(seconds=remaining_batches*mean_time)
		else:
			print()
			print("Training finished.")
		load_model = True

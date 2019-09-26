from environment import Environment

class Agent:
	"""
	Used to implement the A3C algorithm
	"""

	def __init__(self, index):
		"""
		Initializes a new agent

		Parameters
		----------
		index : int
			used to set up the agent name
		"""

		self.name = "agent_{0}".format(index)
		self.local_env = Environment("env_{0}".format(index))

	def train(self, sess, coord, lock):
		"""
		Trains the agent

		Parameters
		----------
		sess : tf.Session()
			Tensorflow session used to run the train
		lock : threading.Lock()
			used to properly load a new fruit by correctly syncing the local environment
		"""

		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():

				lock.acquire()
				self.local_env.load_fruit(sess)
				lock.release()
				
				print(self.local_env.fruit.index)
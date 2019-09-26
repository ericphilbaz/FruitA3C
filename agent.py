from environment import Environment

class Agent:

	def __init__(self, index):

		self.name = "agent_{0}".format(index)
		self.local_env = Environment("env_{0}".format(index))

	def train(self, sess):

		with sess.as_default(), sess.graph.as_default():
			self.local_env.load_fruit(sess)
			print(self.local_env.fruit.index)
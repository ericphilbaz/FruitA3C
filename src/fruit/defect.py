import numpy as np
import math
from src.fruit.utils import log10_trasform, sigmoid
from uuid import uuid4

class Defect:
	"""
	Defect object find on fruits
	"""

	def __init__(self, shot_name, index, shot_sizes, props):
		"""
		Instantiates Defect objects

		Parameters
		----------
		shot_name : str
			name of the shot
		index : int
			index of the defect (on the fruit)
		bounding_box : array
			bounding limits of the defect
		area : int
			area of the defect
		shot_sizes : array
			sizes of the shot
		"""

		self.shot_name = shot_name
		self.shot_sizes = shot_sizes

		self.guesses = []
		self.uuid = None
		self.index = index

		self.area = props.area
		self.perimeter = props.perimeter

		self.y_center, self.x_center = props.centroid
		self.circularity = (4*math.pi*self.area) / (self.perimeter*self.perimeter)
		self.eccentricity = props.eccentricity
		self.solidity = props.solidity
		# self.moments_hu = log10_trasform(props.moments_hu)

	def __eq__(self, defect):
		return True if self.index == defect.index else False

	def __sub__(self, defect):
		"""
		Used to return differences between two defects

		Parameters
		----------
		defect : Defect
			other defect

		Returns
			an array of differences, in absolute value, between 0 (different) and 1 (same)
		"""

		noise = 0.01

		delta_x = 1 - np.abs(self.x_center - defect.x_center)/self.shot_sizes[1] \
				+ noise*(2*np.random.rand()-1)
		delta_y = 1 - np.abs(self.y_center - defect.y_center)/self.shot_sizes[0] \
				+ noise*(2*np.random.rand()-1)

		delta_circularity = 1 - np.abs(self.circularity - defect.circularity) \
				+ noise*(2*np.random.rand()-1)
		delta_eccentricity = 1 - np.abs(self.eccentricity - defect.eccentricity) \
				+ noise*(2*np.random.rand()-1)
		delta_solidity = 1 - np.abs(self.solidity - defect.solidity) \
				+ noise*(2*np.random.rand()-1)
		# delta_hu = 1 - sigmoid(np.linalg.norm(self.moments_hu - defect.moments_hu)) \
		# 		+ noise*(2*np.random.rand()-1)

		delta = [delta_x, delta_y, delta_circularity,
				delta_eccentricity, delta_solidity]
		return np.array(delta).reshape((1, 5))

	def choose_uuid(self):

		if self.guesses:
			self.uuid = max(self.guesses, key=self.guesses.count)
		else:
			self.uuid = uuid4()
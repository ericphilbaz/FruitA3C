import numpy as np
import math

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

		self.uuid = None
		self.index = index

		self.area = props.area
		self.perimeter = props.perimeter
		self.y_center, self.x_center = props.centroid
		self.circularity = (4*math.pi*self.area) / (self.perimeter*self.perimeter)
		self.eccentricity = props.eccentricity
		self.solidity = props.solidity

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

		delta = [delta_x, delta_y, delta_circularity, delta_eccentricity, delta_solidity]
		return np.array(delta).reshape((1, 5))
import numpy as np
import math

class Defect:
	"""
	Defect object find on fruits
	"""

	def __init__(self, shot_name, index, shot_sizes, bounding_box, area, perimeter):
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

		self.x_center = (bounding_box[3] - bounding_box[1])/2
		self.y_center = (bounding_box[2] - bounding_box[0])/2
		# self.area = area
		# self.perimeter = perimeter
		self.circularity = (4*math.pi*area) / (perimeter*perimeter)
		print(self.circularity)

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
		# delta_area = 1 - np.abs(self.area - defect.area)/(self.shot_sizes[1]*self.shot_sizes[0]) \
		# 		+ noise*(2*np.random.rand()-1)
		delta_circularity = 1 - np.abs(self.circularity - defect.circularity) \
				+ noise*(2*np.random.rand()-1)

		return np.array([delta_x, delta_y, delta_circularity]).reshape((1, 3))
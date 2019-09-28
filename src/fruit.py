import tifffile
import ast
from src.defect import Defect
from skimage.measure import label, regionprops

class Fruit:
	"""
	Used to hold and handle Defect objects
	"""

	def __init__(self, index, load_path, defects_thresholds=[160]):
		"""
		Instantiates the Fruit object

		Parameters
		----------
		load_path : str
			load path of the fruit's shots
		defects_thresholds : list
			list of thresholds for labeling
		"""

		self.index = index
		self.defects = Fruit.load(load_path, index, defects_thresholds)

		self.shots_tot = len(self.defects)
		self.defects_tot = sum([len(defects_per_shot) for defects_per_shot in self.defects])

		self.shots_index = next(i for i, defects_list in enumerate(self.defects) if defects_list) if self.defects_tot else 0
		self.defects_in_shots_index = 0
		self.defects_index = -1

		# self.defects_analyzed = 0
		self.defects_identified = 0
		
	def __iter__(self):
		return self

	def __next__(self):
		"""
		Return the next defect to be analyzed and updates all the counters

		Returns
		-------
		defect : Defect
			next defect to be analyzed
		"""

		if self.defects_in_shots_index == len(self.defects[self.shots_index]):
			self.defects_in_shots_index = 0
			self.shots_index += next((i for i, v in enumerate(self.defects[self.shots_index+1:]) if v), 0)+1
			
		if self.defects_index < self.defects_tot-1:
			defect = self.defects[self.shots_index][self.defects_in_shots_index]
			self.defects_in_shots_index += 1
			self.defects_index += 1
			return defect
		else:
			raise StopIteration


	@staticmethod
	def load(load_path, index, defects_thresholds):
		"""
		Loads shots and answers (indices of defects on the fruit)

		Parameters
		----------
		load_path : str
			load path of the fruit's shots
		defects_thresholds : list
			list of thresholds for labeling

		Returns
		-------
		defects : list
			list of defects divided in sublists (shots)
		"""

		name = load_path + "{0}.tiff".format(index)
		with tifffile.TiffFile(name) as tif:
			shots = tif.asarray()
			answers = ast.literal_eval(tif.pages[0].tags["ImageDescription"].value)

		defects = []
		for i, (shot, answers_list) in enumerate(zip(shots, answers)):
			thresholds = shot < defects_thresholds[0]
			labels = label(thresholds)
			defects_in_shot = [Defect("{0}_{1}".format(index, i), answer, defect.bbox, defect.area, shot[i].shape) for answer, defect in zip(answers_list, regionprops(labels))]
			defects.append(defects_in_shot)

		return defects
import tifffile
import ast
from defect import Defect
from skimage.measure import label, regionprops

class Fruit:
	"""
	Used to hold and handle Defect objects
	"""

	def __init__(self, index, load_path, defects_thresholds=[160]):
		"""
		Instantiates the Fruit object
		"""

		self.index = index
		self.defects = Fruit.load(load_path, index, defects_thresholds)

	@staticmethod
	def load(load_path, index, defects_thresholds):

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
from src.fruit import Fruit

f = Fruit(0, "dataset/dataset/")

for d in f:
	print()
	print(f.shots_index, f.shots_tot)
	print(f.defects_index, f.defects_tot)
	print("###", d.shot_name, d.index, "###")
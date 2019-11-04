import src.train as train
import src.test as test
import multiprocessing

load_path = "dataset/dataset2/"
# load_path = "dataset/sample/"
model_path = './model'

# n_agents = 1
n_agents = multiprocessing.cpu_count()

starting_index = 0
final_index = 128
batch = 32
load_model = False

testing_index = 0

def main():
	train.run(n_agents, load_path, model_path, starting_index, final_index, batch, load_model)
	# test.run(load_path, model_path, testing_index, load_model)

if __name__ == "__main__":
	main()
import train
import multiprocessing

load_path = "dataset/dataset/"
# load_path = "dataset/sample/"
model_path = './model'
load_model = True

n_agents = 1
# n_agents = multiprocessing.cpu_count()

starting_index = 64
final_index = 128
batch = 8

def main():
	train.run(n_agents, load_path, model_path, starting_index, final_index, batch, load_model)

if __name__ == "__main__":
	main()
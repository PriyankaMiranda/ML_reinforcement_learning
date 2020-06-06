import numpy as np
import matplotlib.pyplot as plt

class my_data:
	def __init__(self, data_quality, max_data_quality):
		self.data_quality = data_quality # data quality of the dataset
		self.avg_data_quality = 0
		self.N = 0 # nunmber of time the dataset is fetched 

	def get_data(self):
		return np.random.randn() + self.data_quality # gaussian with unit variance

	def update_set_params(self, new_data_val):# latest sample received from the exp
		self.N += 1
		self.avg_data_quality = (1 - 1.0/self.N)*self.avg_data_quality + new_data_val/self.N

class my_data_optimistic:
	def __init__(self, data_quality, max_data_quality):
		self.data_quality = data_quality # data quality of the dataset
		self.avg_data_quality = max_data_quality
		self.N = 1 # nunmber of time the dataset is fetched 

	def get_data(self):
		return np.random.randn() + self.data_quality # gaussian with unit variance

	def update_set_params(self, new_data_val):# latest sample received from the exp
		self.N += 1
		self.avg_data_quality = (1 - 1.0/self.N)*self.avg_data_quality + new_data_val/self.N


def select_best_sets(dataset_labels, num_of_good_sets):
	np.random.shuffle(dataset_labels)
	return list(set(dataset_labels[0:num_of_good_sets+1]))

def create_datasets(N):
	dataset_labels = np.empty(N)
	for x in range(0,N):
		dataset_labels[x] = x+1
	return dataset_labels

def exp(total_num_of_sets, num_of_good_sets, num_of_tries, eps, max_data_quality, min_data_quality, variance):

	dataset_labels = create_datasets(total_num_of_sets)
	data = np.empty(num_of_tries)
	good_dataset = select_best_sets(dataset_labels, num_of_good_sets)
	my_datasets = []
	my_datasets2 = []
	my_datasets3 = []
	my_data_qualities = []
	for x in range(0,total_num_of_sets):
		if (x in good_dataset):
			data_quality = max_data_quality - np.random.randint(low=0, high=variance)
			while(data_quality in my_data_qualities):
				data_quality = max_data_quality - np.random.randint(low=0, high=variance)
		else:
			data_quality = min_data_quality + np.random.randint(low=0, high=variance)
			while(data_quality in my_data_qualities):
				data_quality = min_data_quality + np.random.randint(low=0, high=variance)
		my_data_qualities.append(data_quality)

		my_datasets.append(my_data(data_quality,max_data_quality))
		my_datasets2.append(my_data_optimistic(data_quality,max_data_quality)) 
		my_datasets3.append(my_data_optimistic(data_quality,max_data_quality)) 

	data = np.zeros(shape=(3,num_of_tries)) # 2d array for n different eps trials and the number of tries


	for x in range(total_num_of_sets):
		init_update = my_datasets3[x].get_data()
		my_datasets3[x].update_set_params(init_update)

	for i in range(num_of_tries):
		rand_eps_val = np.random.random()*100
		rand_dataset_selected = np.random.choice(total_num_of_sets)
		if rand_eps_val < eps:
			print("random data selected")
			dataset_selected = rand_dataset_selected
		else:
			print("highest in current selected")
			dataset_selected = np.argmax([curr_set.avg_data_quality for curr_set in my_datasets])
		dataset_selected2 = np.argmax([curr_set.avg_data_quality for curr_set in my_datasets2])
		dataset_selected3 = np.argmax([curr_set.avg_data_quality + np.sqrt(2*np.log(i+(total_num_of_sets)) / curr_set.N) for curr_set in my_datasets3])# -----------------------------------

		data[0][i] = my_datasets[int(dataset_selected)].get_data()
		data[1][i] = my_datasets2[int(dataset_selected2)].get_data()
		data[2][i] = my_datasets2[int(dataset_selected3)].get_data()
		
		my_datasets[int(dataset_selected)].update_set_params(data[0][i])
		my_datasets2[int(dataset_selected2)].update_set_params(data[1][i])
		my_datasets3[int(dataset_selected3)].update_set_params(data[2][i])

	series_avg_accuracy = np.cumsum(data[0]) / (np.arange(num_of_tries) + 1)
	plt.plot(series_avg_accuracy, label= 'epsilon = '+str(eps)+', epsilon greedy')
	series_avg_accuracy = np.cumsum(data[1]) / (np.arange(num_of_tries) + 1)
	plt.plot(series_avg_accuracy, label= 'epsilon = '+str(eps)+', epsilon greedy optimistic')
	series_avg_accuracy = np.cumsum(data[2]) / (np.arange(num_of_tries) + 1)
	plt.plot(series_avg_accuracy, label= 'epsilon = '+str(eps)+', chernoff hoeffding bound')

	for x in range(0,total_num_of_sets):
		plt.plot(np.ones(num_of_tries)*my_data_qualities[x], 'r--')

	plt.legend()
	# plt.xscale('log')
	plt.show()

def print_data_for_user(total_num_of_sets, num_of_good_sets, num_of_tries, eps, max_data_quality, min_data_quality, variance):
	print("Epsilon-greedy algorithm VS  Optimistic initial values algorithm")
	print("-----------------Default values-----------------")
	print("Number of datasets: "+str(total_num_of_sets))
	print("Number of good datasets: "+str(num_of_good_sets))
	print("Number of tries: "+str(num_of_tries))
	print("Maximum data quality: "+str(max_data_quality))
	print("Minimum data quality: "+str(min_data_quality))
	print("Epsilon values: "+str(eps))
	print("------------------------------------------------")

if __name__ == '__main__':
	#---constant params---
	total_num_of_sets = 5
	num_of_good_sets = 1
	max_data_quality = 10
	min_data_quality = 1
	num_of_tries = 100000
	variance = 5 
	eps = 5
	#---------------------
	print_data_for_user(total_num_of_sets, num_of_good_sets,num_of_tries,eps, max_data_quality, min_data_quality, variance)
	exp(total_num_of_sets, num_of_good_sets,num_of_tries,eps, max_data_quality, min_data_quality, variance)

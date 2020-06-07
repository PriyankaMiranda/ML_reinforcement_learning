import numpy as np
import matplotlib.pyplot as plt

class my_data:
	def __init__(self, data_quality, max_data_quality):
		self.data_quality = data_quality # data quality of the dataset
		self.avg_data_quality = max_data_quality
		self.N = 1 # nunmber of time the dataset is fetched 

	def get_data(self):
		print("Data quality : "+str(self.data_quality))	
		return self.data_quality - np.random.randn() # gaussian with unit variance

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
	print("Suppose we have n datasets : a1, a2 ... Default value n = 5.")
	dataset_labels = create_datasets(total_num_of_sets)
	data = np.empty(num_of_tries)
	print("Among these, we randomly assign x of these as the ones that produce the best results. Default value x = 2")
	good_dataset = select_best_sets(dataset_labels, num_of_good_sets)
	print("PS. We know that these datasets "+str(good_dataset)+" are meant to produce the best results.")
	print("Let us observe and understand how the algorithm determines that without using many tries")
	my_datasets = []
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
	
	my_datasets_copies = []
	for x in range(0,len(eps)):
		my_datasets_copies.append(my_datasets)
	
	print("If randomly generated epsilon < eps, We choose randomly among the datasets.")
	print("Else if randomly generated epsilon > eps, We choose the one with the best average data quality")
	print("Number of tries we currently have to make sure we use the best data: "+str(num_of_tries))
	

	dataset_selected = np.empty(len(eps)) # current dataset is selected n different times based on eps values
	data = np.zeros(shape=(len(eps),num_of_tries)) # 2d array for n different eps trials and the number of tries

	for i in range(num_of_tries):
		rand_eps_val = np.random.random()*100

		rand_dataset_selected = np.random.choice(len(eps))
		# print("Epsilon (ε) : "+str(rand_eps_val))
		for x in range(0,len(eps)):
			dataset_selected[x] = np.argmax([curr_set.avg_data_quality for curr_set in my_datasets_copies[x]])
			data[x][i] = my_datasets_copies[x][int(dataset_selected[x])].get_data()
			my_datasets_copies[x][int(dataset_selected[x])].update_set_params(data[x][i])


	for x in range(0,len(eps)):		
		series_avg_accuracy = np.cumsum(data[x]) / (np.arange(num_of_tries) + 1)
		# plot updated average accuracy as we explore datasets
		plt.plot(series_avg_accuracy, label= 'epsilon = '+str(eps[x]))

	for x in range(0,total_num_of_sets):
		plt.plot(np.ones(num_of_tries)*my_data_qualities[x], 'r--')

	plt.legend()
	# plt.xscale('log')
	plt.show()




def print_data_for_user(total_num_of_sets, num_of_good_sets, num_of_tries, eps, max_data_quality, min_data_quality, variance):
	print("A simple experiment demonstrating epsilon-greedy, optimistic initial value algorithm")
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
	num_of_tries = 1000
	variance = 5
	#---------------------

	#----varying parms---- 
	eps = [90,50,20,10,5]
	#---------------------
	print_data_for_user(total_num_of_sets, num_of_good_sets,num_of_tries,eps, max_data_quality, min_data_quality, variance)
	exp(total_num_of_sets, num_of_good_sets,num_of_tries,eps, max_data_quality, min_data_quality, variance)

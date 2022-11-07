from .requests import *
import random


class Dataset:
	def __init__(self, max_n_vnfs:int):
		self.info = 'not defined'
		self.payload = []
		self.max_n_vnfs = max_n_vnfs

	def set_terminal(self) -> None:
		'''
		sets the last SFC of the payload as terminal
		'''
		self.payload[-1].is_terminal = True 

	def __str__(self):
		'''
		custom print method
		'''
		string = ''
		for sfc in self.payload:
			string += str(sfc)
		return '{}: \n{}'.format(self.info, string)


##############################################################
# Dataset generators

def dummy_payload(n_sfcs:int, min_n_vnfs: int, max_n_vnfs:int):
	'''
	creates a dataset with SFCs comprising random VNFs
	'''
	dummy = Dataset(max_n_vnfs)
	dummy.info = 'Dummy dataset for preliminary testing with {} SFCs'.format(n_sfcs)

	for id in range(1, n_sfcs+1):
		vnf_list = []
		order_list = range(1, random.randint(min_n_vnfs+1, max_n_vnfs)+1)

		for order in order_list:
			vnf = RandomVNF(order)
			vnf_list.append(vnf)

		INGRESS = random.randint(1, 10) / 100
		LATENCY = random.randint(50, 200) / 200
		LIFESPAN = random.randint(10, 100) / 100
		ARRIVAL = random.randint(1, 100)
		SRC = random.choice([(0.1, 0.4), (0.7, 0.2), (0.5, 0.5), (0.1, 0.6), (0.9, 0.7),
							 (0.1, 0.3), (0.7, 0.6), (0.3, 0.5), (0.2, 0.6), (0.4, 0.7)])
		DST = random.choice([(0.1, 0.4), (0.7, 0.2), (0.5, 0.5), (0.1, 0.6), (0.9, 0.7),
							 (0.1, 0.3), (0.7, 0.6), (0.3, 0.5), (0.2, 0.6), (0.4, 0.7)])

		dummy_SFC = SFC(id, INGRESS, vnf_list, LATENCY, LIFESPAN, ARRIVAL, SRC, DST)
		dummy_SFC.dataset = dummy # assign the dataset to the SFC 
		dummy_SFC.config() 
		dummy.payload.append(dummy_SFC) 

	return dummy

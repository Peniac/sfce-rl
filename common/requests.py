from typing import List, Tuple
import networkx as nx 
import random


class Request:
	def __init__(self):
		self.sfc = None
		self.demand = None
		self.assigned_to = None

	def remove(self) -> None:
		'''
		removes the request from the resource that is hosting it
		'''
		resource = self.assigned_to
		# links are not assigned to anything yet 
		if resource is None:
			pass 
		else:
			resource.remove(self)


class VNF(Request):
	def __init__(self, order: int):
		super().__init__()
		self.order = order
		self.ingress = None
		self.egress = None
		self.is_first = False # used for the DummyIn
		self.is_terminal = False # used for the DummyOut 

		self.is_last = False # used for the last (non-Dummy) VNF of the chain (to terminate an episode)

	# def get_state(self) -> VnfState:
	# 	'''
	# 	returns the VNF state
	# 	'''
	#
	# 	state = VnfState(
	# 		vnf_demand = self.demand,
	# 		vnf_ingress = self.ingress,
	# 		vnf_egress = self.egress,
	# 		vnf_order = (self.order) / (self.sfc.length)
	# 		)
	#
	# 	return state


class DummyIn(VNF):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.is_first = True

	def profile(self, traffic_load) -> None:
		'''
		sets resource requirements according to traffic load
		'''
		self.ingress = 0
		self.egress = traffic_load
		self.demand = 0
		self.longitude = self.sfc.src[0]
		self.latitude = self.sfc.src[1]


class DummyOut(VNF):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.is_terminal = True

	def profile(self, traffic_load) -> None:
		'''
		sets resource requirements according to traffic load
		'''
		self.ingress = traffic_load
		self.egress = 0
		self.demand = 0
		self.longitude = self.sfc.dst[0]
		self.latitude = self.sfc.dst[1]	


class RandomVNF(VNF):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def profile(self, traffic_load) -> None:
		'''
		sets resource requirements according to traffic load
		'''

		self.ingress = traffic_load
		self.egress = random.uniform(0.7, 1) * self.ingress
		self.demand = round(random.uniform(0.05, 0.2),2) # 5-20% of a server's CPU
		#self.demand = 0.2

class VLink(Request):
	def __init__(self, VNF1: VNF, VNF2: VNF, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.VNF1 = VNF1
		self.VNF2 = VNF2
		self.demand = VNF1.egress


class SFC():
	def __init__(self, id: int, ingress: float, vnfs: List[VNF], 
		latency: float, lifespan: float, arrival: int, 
		src: Tuple[float,float], dst: Tuple[float,float]):
		self.id = id
		self.ingress = ingress
		self.vnfs = vnfs
		self.length = len(self.vnfs)
		self.latency = latency
		self.lifespan = lifespan
		self.clock = lifespan
		self.arrival = arrival
		self.src = src
		self.dst = dst
		self.G = nx.DiGraph()
		self.is_terminal = False
		self.dataset = None
		self.placed_optimal = False
		
	def config(self) -> None:
		'''
		inserts dummy in and out nodes, creates the VNF graph, and sets VNF and VLink resource requirements
		'''

		IN = DummyIn(order=0)
		OUT = DummyOut(order=len(self.vnfs)+1)
		self.vnfs += [IN,OUT]
		self.vnfs = sorted(self.vnfs, key=lambda x: x.order)
		self.vlinks = []

		for vnf in self.vnfs:
			vnf.sfc = self # assign the SFC object to each VNF

		for vnf_i in self.vnfs:

			if vnf_i.order == 0:
				vnf_i.profile(self.ingress)

			for vnf_j in self.vnfs:

				if vnf_j.order == vnf_i.order + 1: 
					vnf_j.profile(vnf_i.egress)
					self.G.add_edge(vnf_i,vnf_j) 
					self.G.edges[vnf_i, vnf_j]['VLink'] = VLink(vnf_i, vnf_j)
					self.vlinks.append(VLink(vnf_i, vnf_j))
					break # one egress edge per VNF

		self.vnfs[-2].is_last = True # set the last non-Dummy VNF of the chain as "last"

	# def get_state(self) -> SfcState:
	# 	'''
	# 	returns the state of the current SFC
	# 	'''
	#
	# 	state = SfcState(
	# 		sfc_length = self.length / self.dataset.max_n_vnfs,
	# 		sfc_src_longitude = self.src[0],
	# 		sfc_src_latitude = self.src[1],
	# 		sfc_dst_longitude = self.dst[0],
	# 		sfc_dst_latitude = self.dst[1]
	# 		)
	#
	# 	return state

	def __str__(self):
		'''
		custom print method
		'''
		
		string = ["VNF{}({})".format(vnf.order,vnf.demand) for vnf in self.vnfs]
		return "SFC{}={} \n".format(self.id, string)

	def remove(self) -> None:
		'''
		removes VNFs and VLinks from physical resources
		'''
		for vnf in self.vnfs:
			vnf.remove()
		for vlink in self.vlinks:
			vlink.remove()
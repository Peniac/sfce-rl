import pickle
import os

import networkx as nx
import numpy as np

from gym.spaces import Box, Discrete, Dict


def save_object(obj, filename):
    cwd = os.getcwd()
    filepath = os.path.join(cwd, 'results', filename)

    with open(filepath, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_observation_space_per_agent(topology):
    """
    Currently only 'fingerprint' is missing.
    :param topology:
    :return:
    """
    local_observation = {'in_shortest_path': Discrete(2),
                         'hosts_previous': Discrete(2),
                         'hosts_another': Discrete(2),
                         'longitude': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                         'latitude': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)}

    # Create one observation input for each server.
    pops = list(topology.G.nodes)
    pop = pops[0]
    for idx, s_capacity in enumerate(pop.s_capacities):
        local_observation['server' + str(idx)] = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    sfc_observation = {'sfc_length': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                       'sfc_src_longitude': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                       'sfc_src_latitude': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                       'sfc_dst_longitude': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                       'sfc_dst_latitude': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)}

    vnf_observation = {'vnf_demand': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                       'vnf_ingress': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                       'vnf_egress': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                       'vnf_order': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)}

    # Create one observation input for each neighbor.
    neighbor_observation = {}
    neighbors = [p for p in topology.G.nodes if p != pop]
    for idx, n in enumerate(neighbors):
        neighbor_observation['capacity' + str(idx)] = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        neighbor_observation['longitude' + str(idx)] = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        neighbor_observation['latitude' + str(idx)] = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    observation = {}
    for d in [local_observation, sfc_observation, vnf_observation, neighbor_observation]:
        observation.update(d)

    return Dict({'obs': Dict(observation)})


def compute_flags(topology=None,
                  pop=None,
                  vnf=None,
                  sfc=None):

    in_shortest_path = _in_shortest_path(topology=topology,
                                         pop=pop,
                                         sfc=sfc)

    hosts_previous = _hosts_previous(pop=pop,
                                     vnf=vnf,
                                     sfc=sfc)
    hosts_another = 0

    return in_shortest_path, hosts_previous, hosts_another


def _hosts_previous(pop=None,
                    vnf=None,
                    sfc=None):
    hosts_previous = (1 if sfc.vnfs[vnf.order-1].assigned_to == pop else 0)

    return hosts_previous


def _in_shortest_path(topology=None,
                      pop=None,
                      sfc=None):
    src = sfc.vnfs[0].assigned_to
    dst = sfc.vnfs[-1].assigned_to
    shortest_paths = nx.all_shortest_paths(topology.G, source=src, target=dst)
    in_shortest_path = any([pop in path for path in shortest_paths])

    return in_shortest_path


def moving_average(my_list):
    avg = [my_list[0]]
    curr_sum = my_list[0]

    for i in range(1, len(my_list)):
        curr_sum += my_list[i]
        avg.append(curr_sum / i)

    return avg

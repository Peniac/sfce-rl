from typing import List, Dict, Tuple
from collections import deque, defaultdict
import networkx as nx
import numpy as np
import random

random.seed(123)


class Resource:
    def __init__(self, id: int, capacity: float):
        self.id = id
        self.init_capacity = capacity
        self.capacity = capacity
        self.allocated_to = []  # stores requests assigned to the resource

    def can_fit(self, request) -> bool:
        '''
        checks whether a request fits into the resource
        '''

        flag = self.capacity > request.demand
        return flag

    def allocate_to(self, request) -> None:
        '''
        populates the "allocated_to" list and updates the residual resources
        '''

        self.allocated_to.append(request)
        request.assigned_to = self
        self.capacity -= request.demand

    def remove(self, request) -> None:
        '''
        removes the request from the resource and updates the history
        '''

        self.allocated_to.remove(request)
        self.capacity += request.demand

    def reset(self) -> None:
        '''
        sets available resources to random and empties the "allocated_to" list
        '''

        self.capacity = round(random.uniform(0.5, 1.0), 2)
        self.allocated_to.clear()


class PoP(Resource):
    def __init__(self, id: int, capacity: float, longitude: float, latitude: float, n_servers: int):
        super().__init__(id, capacity)
        self.longitude = longitude
        self.latitude = latitude

        # new in version 0.3
        self.n_servers = n_servers
        self.s_capacities = []
        self.l_capacities = []

    def can_fit(self, request) -> bool:
        '''
        checks whether a request fits into the resource
        '''

        flag = any(i >= request.demand for i in self.s_capacities)
        return flag

    def allocate_to(self, request) -> None:
        '''
        populates the "allocated_to" list and updates the residual resources
        '''

        self.allocated_to.append(request)
        request.assigned_to = self

        argmax = np.argmax(self.s_capacities)
        self.s_capacities[argmax] -= request.demand
        self.capacity -= request.demand

    def reset(self) -> None:
        '''
        resets the capacity of servers
        '''

        self.s_capacities = [round(random.uniform(0.0, 0.3), 2) for _ in range(self.n_servers)]
        self.capacity = sum(self.s_capacities)

        self.allocated_to.clear()

    def remove(self, request):
        '''
        removes the request from the resource
        '''

        self.allocated_to.remove(request)

        argmin = np.argmin(self.s_capacities)
        self.s_capacities[argmin] += request.demand
        self.capacity += request.demand

    # def get_local_state(self) -> List[float]:
    #     '''returns the local state'''
    #
    #     state = LocalState(
    #         server_capacities=self.s_capacities,
    #         link_capacities=self.l_capacities,
    #         longitude=self.longitude,
    #         latitude=self.latitude,
    #         hosts_another=0,
    #         hosts_previous=0,
    #         in_shortest_path=0
    #     )
    #
    #     return state

    # def get_condensed_state(self) -> float:
    #     '''returns the condensed state'''
    #
    #     state = CondensedState(
    #         capacity=self.capacity / self.n_servers,
    #         longitude=self.longitude,
    #         latitude=self.latitude
    #     )
    #
    #     return state


class Link(Resource):
    def __init__(self, id: int, capacity: float, PoP1: PoP, PoP2: PoP):
        super().__init__(id, capacity)
        self.PoP1 = PoP1
        self.PoP2 = PoP2


class Topology:
    def __init__(self, topo_type, filename=None, *args):
        self.G = nx.Graph()
        self.temp_node_assignments = {}
        self.temp_link_assignments = defaultdict(list)
        self.hosted_sfcs = []

        self.type = topo_type

        if self.type == 'MESH':
            PoP1 = PoP(id=0, capacity=1, longitude=0.1, latitude=0.4, n_servers=10)
            PoP2 = PoP(id=1, capacity=1, longitude=0.7, latitude=0.2, n_servers=10)
            PoP3 = PoP(id=2, capacity=1, longitude=0.5, latitude=0.5, n_servers=10)
            PoP4 = PoP(id=3, capacity=1, longitude=0.1, latitude=0.6, n_servers=10)
            PoP5 = PoP(id=4, capacity=1, longitude=0.9, latitude=0.7, n_servers=10)

            Link12 = Link(id=0, capacity=1, PoP1=PoP1, PoP2=PoP2)
            Link13 = Link(id=1, capacity=1, PoP1=PoP1, PoP2=PoP3)

            self.G.add_edge(PoP1, PoP2, Link=Link12)
            self.G.add_edge(PoP1, PoP3, Link=Link13)

            Link23 = Link(id=2, capacity=1, PoP1=PoP2, PoP2=PoP3)
            Link25 = Link(id=3, capacity=1, PoP1=PoP2, PoP2=PoP5)

            self.G.add_edge(PoP2, PoP3, Link=Link23)
            self.G.add_edge(PoP2, PoP5, Link=Link25)

            Link34 = Link(id=4, capacity=1, PoP1=PoP3, PoP2=PoP4)

            self.G.add_edge(PoP3, PoP4, Link=Link34)

        elif self.type == 'MESH_LARGE':
            PoP1 = PoP(id=0, capacity=1, longitude=0.1, latitude=0.4, n_servers=10)
            PoP2 = PoP(id=1, capacity=1, longitude=0.7, latitude=0.2, n_servers=10)
            PoP3 = PoP(id=2, capacity=1, longitude=0.5, latitude=0.5, n_servers=10)
            PoP4 = PoP(id=3, capacity=1, longitude=0.1, latitude=0.6, n_servers=10)
            PoP5 = PoP(id=4, capacity=1, longitude=0.9, latitude=0.7, n_servers=10)

            PoP6 = PoP(id=5, capacity=1, longitude=0.1, latitude=0.3, n_servers=10)
            PoP7 = PoP(id=6, capacity=1, longitude=0.7, latitude=0.6, n_servers=10)
            PoP8 = PoP(id=7, capacity=1, longitude=0.3, latitude=0.5, n_servers=10)
            PoP9 = PoP(id=8, capacity=1, longitude=0.2, latitude=0.6, n_servers=10)
            PoP10 = PoP(id=9, capacity=1, longitude=0.4, latitude=0.7, n_servers=10)

            Link12 = Link(id=0, capacity=1, PoP1=PoP1, PoP2=PoP2)
            Link13 = Link(id=1, capacity=1, PoP1=PoP1, PoP2=PoP3)
            self.G.add_edge(PoP1, PoP2, Link=Link12)
            self.G.add_edge(PoP1, PoP3, Link=Link13)

            Link23 = Link(id=2, capacity=1, PoP1=PoP2, PoP2=PoP3)
            Link25 = Link(id=3, capacity=1, PoP1=PoP2, PoP2=PoP5)
            self.G.add_edge(PoP2, PoP3, Link=Link23)
            self.G.add_edge(PoP2, PoP5, Link=Link25)

            Link34 = Link(id=4, capacity=1, PoP1=PoP3, PoP2=PoP4)
            self.G.add_edge(PoP3, PoP4, Link=Link34)

            # Link PoP 5 with PoP 6
            Link56 = Link(id=5, capacity=1, PoP1=PoP5, PoP2=PoP6)
            self.G.add_edge(PoP5, PoP6, Link=Link56)

            Link67 = Link(id=6, capacity=1, PoP1=PoP6, PoP2=PoP7)
            Link68 = Link(id=7, capacity=1, PoP1=PoP6, PoP2=PoP8)
            self.G.add_edge(PoP6, PoP7, Link=Link67)
            self.G.add_edge(PoP6, PoP8, Link=Link68)

            Link78 = Link(id=8, capacity=1, PoP1=PoP7, PoP2=PoP8)
            Link710 = Link(id=9, capacity=1, PoP1=PoP7, PoP2=PoP10)
            self.G.add_edge(PoP7, PoP8, Link=Link78)
            self.G.add_edge(PoP7, PoP10, Link=Link710)

            Link910 = Link(id=10, capacity=1, PoP1=PoP9, PoP2=PoP10)
            self.G.add_edge(PoP9, PoP10, Link=Link910)

        else:
            raise NotImplementedError

    def __str__(self):
        '''
        custom print method
        '''

        nodes = [node.id for node in self.G.nodes]
        edges = [(self.G.edges[edge]['Link'].PoP1.id, self.G.edges[edge]['Link'].PoP2.id) for edge in self.G.edges]
        string = 'Nodes: {}\n'.format(nodes)
        string += 'Edges: {}\n'.format(edges)
        return string

    def reset(self) -> None:
        '''
        resets capacities of resources (triggered at the start of each episode)
        '''

        for PoP in self.G.nodes:
            PoP.reset()
        for e in self.G.edges:
            Link = self.G.edges[e]['Link']
            Link.reset()

        # populate the link capacities list of PoPs
        for PoP in self.G.nodes:
            PoP.l_capacities = []
            links = list(self.G.edges(PoP))
            for link in links:
                PoP.l_capacities.append(self.G.edges[link]['Link'].capacity)

    def node_assignment(self, vnf_solution, vnf) -> bool:
        '''
        places a VNF into a PoP
        '''

        PoP_id = np.argwhere(vnf_solution == 1)[0][0]

        for PoP in self.G.nodes:
            if PoP.id == PoP_id:
                candidate_PoP = PoP

        if candidate_PoP.can_fit(vnf):
            candidate_PoP.allocate_to(vnf)
            self.temp_node_assignments[vnf] = candidate_PoP
            success = True

        else:
            success = False

        return success

    def heuristic_link_assignment(self, sfc) -> Tuple[bool, int, int]:
        '''
        places the virtual links of the SFC
        '''

        success = True
        hop_count = 0

        # for each pair of consecutive VNFs
        for vnf_i, vnf_j in zip(sfc.vnfs, sfc.vnfs[1:]):

            # (i) find the shortest path...
            PoP1 = vnf_i.assigned_to
            PoP2 = vnf_j.assigned_to
            shortest_path = nx.shortest_path(self.G, source=PoP1, target=PoP2, weight=None)

            if vnf_i.is_first:
                src_PoP = PoP1
            if vnf_j.is_terminal:
                dst_PoP = PoP2

            if len(shortest_path) == 1:
                continue

            else:
                shortest_path_links = [self.G.edges[i, j]['Link'] for i, j in zip(shortest_path, shortest_path[1:])]

                # (ii) ... and try to place the respective VLink. If infeasible, return False
                vlink = sfc.G.edges[vnf_i, vnf_j]['VLink']
                for link in shortest_path_links:

                    if link.can_fit(vlink):
                        link.allocate_to(vlink)
                        self.temp_link_assignments[vlink].append(link)
                    else:
                        success = False
                        return success, None, None

                # (iii) compute the length of the path
                hop_count += len(shortest_path) - 1

        optimal_hop_count = nx.shortest_path_length(self.G, source=src_PoP, target=dst_PoP, weight=None)

        return success, hop_count, optimal_hop_count

    def cancel_temp_assignments(self, success):
        '''
        clears the dicts that store temporary assignments
        '''

        if not success:
            _ = [resource.remove(request) for request, resource in self.temp_node_assignments.items()]
            _ = [resource.remove(request) for request in self.temp_link_assignments.keys() for resource in
                 self.temp_link_assignments[request]]
        self.temp_node_assignments.clear()
        self.temp_link_assignments.clear()

    def _expired_sfcs(self):
        expired_sfcs = []

        for sfc in self.hosted_sfcs:
            if sfc.clock < 0:
                expired_sfcs.append(sfc)

        return expired_sfcs

    def remove_expired_sfcs(self):
        expired_sfcs = self._expired_sfcs()

        for sfc in expired_sfcs:
            sfc.remove()
            self.hosted_sfcs.remove(sfc)

    def update_sfc_clock(self):
        for sfc in self.hosted_sfcs:
            sfc.clock -= 1
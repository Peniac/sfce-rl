import numpy as np
import random
from typing import List
from gym.spaces import Discrete, Dict
from collections import OrderedDict

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from utils import get_observation_space_per_agent, compute_flags


class MultiAgentSfcPartitioningEnv(MultiAgentEnv):
    """
    This class implements a multi-agent environment for distributed SFC embedding.
    Assumptions:
        1) SFCs need to be partitioned over a single administrative domain, thus agents are allowed to exchange
        information.
        2) Each PoP of the topology is associated with an RL agent. Each PoP can communicate with its immediate
        neighbors.
        3) The training episodes are part of the environment, i.e., the "dataset" variable.
    """
    def __init__(self, env_config):
        super().__init__()
        self.topology = env_config['topology']
        num_pops = len(self.topology.G.nodes)

        # Define the agent ids.
        self._agent_ids = set(range(num_pops))

        # Define the action space of each agent: {0: low, 1: medium, 2: high}
        self.action_space = Dict({agent_id: Discrete(3) for agent_id in self._agent_ids})

        # Define the observation space of each agent.
        # I think that currently Q-mix supports homogeneous agents only. So I will tweak the obs space to be the same
        # for every agent.
        # self.observation_space = Dict({agent_id: Discrete(2) for agent_id in self._agent_ids})
        self.observation_space = Dict({agent_id: get_observation_space_per_agent(self.topology)
                                       for agent_id in self._agent_ids})

        self._spaces_in_preferred_format = True

        # Initialize evaluation environment
        self.sfc_dataset = env_config['dataset']
        self.sfc_index = -1
        self.current_sfc = self.sfc_dataset.payload[self.sfc_index]
        self.sfc_assignment_encoding = []
        self.vnf_index = 1
        self.current_vnf = self.current_sfc.vnfs[self.vnf_index]

    def reset(self):
        # reset the resources
        self.topology.reset()

        # move to the next SFC of the training dataset
        self.sfc_index += 1
        self.current_sfc = self.sfc_dataset.payload[self.sfc_index]
        # and start with its first VNF
        self.vnf_index = 1
        self.current_vnf = self.current_sfc.vnfs[self.vnf_index]

        # Place the source and destination of the SFC.
        self.place_src_dst()

        state = {}
        for agent_id in self._agent_ids:
            # state[agent_id] = random.choice([0, 1])
            state[agent_id] = self.get_observation_per_agent(agent_id=agent_id)
        print(f'Resetting to state {state}')
        return state

    def next_vnf(self):
        # move to the next VNF
        self.vnf_index += 1
        self.current_vnf = self.current_sfc.vnfs[self.vnf_index]

    def step(self, actions):
        print(f'Current sfc: {self.current_sfc}, Current vnf: {self.current_vnf.order}')
        # Flag that marks the termination of an episode.
        done = False

        action = self.compile_local_actions(actions)

        # Encode the action (similar to one-hot encoding).
        vnf_assignment_encoding = self.encode_action(action)

        # Apply the solution, check feasibility and compute quality.
        node_success = self.topology.node_assignment(vnf_assignment_encoding, self.current_vnf)

        if node_success and self.current_vnf.is_last:
            # link_success, hop_count, optimal_hop_count = self.topology.heuristic_link_assignment(self.current_sfc)
            # if link_success:
            #     self.topology.hosted_sfcs.append(self.current_sfc)
            #     reward = 10 * (1 + optimal_hop_count) / (1 + hop_count)
            # else:
            #     reward = -10
            reward = 10

            # self.topology.cancel_temp_assignments(link_success)

        elif node_success:
            reward = 0.1

        else:
            # self.topology.cancel_temp_assignments(node_success)
            reward = -10

        # move to next state
        if self.current_vnf.is_last or not node_success:
            done = True
        else:
            self.next_vnf()

        state_, rewards, dones, info = {}, {}, {}, {}

        for agent_id in self._agent_ids:
            # state_[agent_id] = random.choice([0, 1])
            state_[agent_id] = self.get_observation_per_agent(agent_id=agent_id)
            rewards[agent_id] = reward
            dones[agent_id] = done
        dones['__all__'] = done

        print(f'next observations are {state_}')
        print(f'rewards are {rewards}')
        print(f'dones are {dones}')
        return state_, rewards, dones, info

    def encode_action(self, action):
        encoding = np.zeros(len(self.topology.G.nodes))
        encoding[action] = 1

        return encoding

    @staticmethod
    def compile_local_actions(actions: dict):
        """
        This function implements the action coordination logic.
        :param actions: 'agent_id, action' dictionary
        :return:
        """
        # Gets key with max value, ties broken arbitrarily.
        chosen_agent_id = max(actions, key=actions.get)
        print(f'chosen agent is {chosen_agent_id}')
        return chosen_agent_id

    def place_src_dst(self):
        sfc = self.current_sfc

        # Place the DummyIn node.
        action = [PoP.id for PoP in self.topology.G.nodes
                  if (PoP.longitude, PoP.latitude) == (sfc.src[0], sfc.src[1])][0]
        encoding = self.encode_action(action)
        _ = self.topology.node_assignment(encoding, sfc.vnfs[0])

        # Place the DummyOut node.
        action = [PoP.id for PoP in self.topology.G.nodes
                  if (PoP.longitude, PoP.latitude) == (sfc.dst[0], sfc.dst[1])][0]
        encoding = self.encode_action(action)
        _ = self.topology.node_assignment(encoding, sfc.vnfs[-1])

    def get_observation_per_agent(self,
                                  agent_id: int = None):
        pop = list(self.topology.G.nodes)[agent_id]
        in_shortest_path, hosts_previous, hosts_another = compute_flags(topology=self.topology,
                                                                        pop=pop,
                                                                        vnf=self.current_vnf,
                                                                        sfc=self.current_sfc)
        local_observation = {'in_shortest_path': int(in_shortest_path),
                             'hosts_previous': int(hosts_previous),
                             'hosts_another': int(hosts_another),
                             'longitude': np.array(pop.longitude, dtype=np.float32),
                             'latitude': np.array(pop.latitude, dtype=np.float32)}

        # Create one observation input for each server.
        for idx, s_capacity in enumerate(pop.s_capacities):
            local_observation['server' + str(idx)] = np.array(s_capacity, dtype=np.float32)

        sfc_observation = {'sfc_length': np.array(self.current_sfc.length / self.current_sfc.dataset.max_n_vnfs, dtype=np.float32),
                           'sfc_src_longitude': np.array(self.current_sfc.src[0], dtype=np.float32),
                           'sfc_src_latitude': np.array(self.current_sfc.src[1], dtype=np.float32),
                           'sfc_dst_longitude': np.array(self.current_sfc.dst[0], dtype=np.float32),
                           'sfc_dst_latitude': np.array(self.current_sfc.dst[1], dtype=np.float32)}

        vnf_observation = {'vnf_demand': np.array(self.current_vnf.demand, dtype=np.float32),
                           'vnf_ingress': np.array(self.current_vnf.ingress, dtype=np.float32),
                           'vnf_egress': np.array(self.current_vnf.egress, dtype=np.float32),
                           'vnf_order': np.array(self.current_vnf.order / self.current_sfc.length, dtype=np.float32)}

        # Create one observation input for each neighbor.
        neighbor_observation = {}
        neighbors = [p for p in self.topology.G.nodes if p != pop]
        for idx, n in enumerate(neighbors):
            neighbor_observation['capacity' + str(idx)] = np.array(sum(n.s_capacities) / len(n.s_capacities), dtype=np.float32)
            neighbor_observation['longitude' + str(idx)] = np.array(n.longitude, dtype=np.float32)
            neighbor_observation['latitude' + str(idx)] = np.array(n.latitude, dtype=np.float32)

        observation = {}
        for d in [local_observation, sfc_observation, vnf_observation, neighbor_observation]:
            observation.update(d)

        for k, v in observation.items():
            if not isinstance(v, int):
                print(v.dtype)
        return OrderedDict({'obs': OrderedDict(observation)})

    # def get_observation_per_agent(self,
    #                               agent_id: int = None):
    #     pop = list(self.topology.G.nodes)[agent_id]
    #     in_shortest_path, hosts_previous, hosts_another = compute_flags(topology=self.topology,
    #                                                                     pop=pop,
    #                                                                     vnf=self.current_vnf,
    #                                                                     sfc=self.current_sfc)
    #     local_observation = {'in_shortest_path': int(in_shortest_path),
    #                          'hosts_previous': int(hosts_previous),
    #                          'hosts_another': int(hosts_another),
    #                          'longitude': pop.longitude,
    #                          'latitude': pop.latitude}
    #
    #     # Create one observation input for each server.
    #     for idx, s_capacity in enumerate(pop.s_capacities):
    #         local_observation['server' + str(idx)] = s_capacity
    #
    #     sfc_observation = {'sfc_length': self.current_sfc.length / self.current_sfc.dataset.max_n_vnfs,
    #                        'sfc_src_longitude': self.current_sfc.src[0],
    #                        'sfc_src_latitude': self.current_sfc.src[1],
    #                        'sfc_dst_longitude': self.current_sfc.dst[0],
    #                        'sfc_dst_latitude': self.current_sfc.dst[1]}
    #
    #     vnf_observation = {'vnf_demand': self.current_vnf.demand,
    #                        'vnf_ingress': self.current_vnf.ingress,
    #                        'vnf_egress': self.current_vnf.egress,
    #                        'vnf_order': self.current_vnf.order / self.current_sfc.length}
    #
    #     # Create one observation input for each neighbor.
    #     neighbor_observation = {}
    #     neighbors = [p for p in self.topology.G.nodes if p != pop]
    #     for idx, n in enumerate(neighbors):
    #         neighbor_observation['capacity' + str(idx)] = sum(n.s_capacities) / len(n.s_capacities)
    #         neighbor_observation['longitude' + str(idx)] = n.longitude
    #         neighbor_observation['latitude' + str(idx)] = n.latitude
    #
    #     observation = {}
    #     for d in [local_observation, sfc_observation, vnf_observation, neighbor_observation]:
    #         observation.update(d)
    #
    #     for k, v in observation.items():
    #         print(type(v))
    #     return OrderedDict({'obs': OrderedDict(observation)})
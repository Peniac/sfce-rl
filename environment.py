import numpy as np
import random
from typing import List
from gym.spaces import Discrete, Box, Dict, MultiDiscrete, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from utils import get_observation_space_per_agent


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

        state = {}
        for agent_id in self._agent_ids:
            state[agent_id] = random.choice([0, 1])
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
            state_[agent_id] = random.choice([0, 1])
            rewards[agent_id] = reward
            dones[agent_id] = done
        dones['__all__'] = done

        # This is what it should look like at the end
        # for agent in self._agent_ids:
        #     state_[agent_id] = self.get_agent_state(agent=agent_id)
        #     rewards[agent_id] = reward # because they share the same reward
        #     dones[agent_id] = done
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

    def place_src_dst(self, sfc):
        # place the DummyIn
        action = [PoP.id for PoP in self.topology.G.nodes
                  if (PoP.longitude, PoP.latitude) == (sfc.src[0], sfc.src[1])][0]
        encoding = self.encode_action(action)
        _ = self.topology.node_assignment(encoding, sfc.vnfs[0])

        # place the DummyOut
        action = [PoP.id for PoP in self.topology.G.nodes
                  if (PoP.longitude, PoP.latitude) == (sfc.dst[0], sfc.dst[1])][0]
        encoding = self.encode_action(action)
        _ = self.topology.node_assignment(encoding, sfc.vnfs[-1])

    def get_shared_state(self):
        raise NotImplementedError

    def get_agent_state(self):
        raise NotImplementedError

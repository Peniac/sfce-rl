import pickle
import os
import numpy as np
from pydantic import BaseModel
from typing import List, Optional


def save_object(obj, filename):
    cwd = os.getcwd()
    filepath = os.path.join(cwd, 'results/objects', filename)

    with open(filepath, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# class LocalState(BaseModel):
#     server_capacities: List[float]
#     link_capacities: List[float]
#     longitude: float
#     latitude: float
#     hosts_another: Optional[bool]
#     hosts_previous: Optional[bool]
#     in_shortest_path: Optional[bool]
#
#
# class CondensedState(BaseModel):
#     capacity: float
#     longitude: float
#     latitude: float
#
#
# class SfcState(BaseModel):
#     # current sfc state
#     sfc_length: float
#     sfc_src_longitude: float
#     sfc_src_latitude: float
#     sfc_dst_longitude: float
#     sfc_dst_latitude: float
#
#
# class VnfState(BaseModel):
#     # current vnf state
#     vnf_demand: float
#     vnf_ingress: float
#     vnf_egress: float
#     vnf_order: float
#
#
# class SharedState(BaseModel):
#     vnf_state: VnfState
#     sfc_state: SfcState
#
#
# class Fingerprint(BaseModel):
#     epsilon: float
#
#
# class FullLocalState(BaseModel):
#     shared_state: SharedState
#     local_state: LocalState
#     condensed_states: List[CondensedState]
#     fingerprint: Fingerprint
#
#     def to_numpy(self):
#         shared_state_list = list(self.shared_state.vnf_state.dict().values()) + \
#                             list(self.shared_state.sfc_state.dict().values())
#
#         local_state_list = self.local_state.server_capacities + \
#                            self.local_state.link_capacities + \
#                            [self.local_state.longitude, self.local_state.latitude] + \
#                            [self.local_state.hosts_another, \
#                             self.local_state.hosts_previous, \
#                             self.local_state.in_shortest_path]
#
#         condensed_state_list = [i for cds in self.condensed_states for i in cds.dict().values()]
#
#         fingerprint_list = list(self.fingerprint.dict().values())
#
#         full_local_state_list = shared_state_list + local_state_list + condensed_state_list + fingerprint_list
#
#         return np.array(full_local_state_list)


def moving_average(mylist):
    avg = [mylist[0]]
    curr_sum = mylist[0]

    for i in range(1, len(mylist)):
        curr_sum += mylist[i]
        avg.append(curr_sum / i)

    return avg
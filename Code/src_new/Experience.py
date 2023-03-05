from Vehicle import Vehicle
from Action import Action

from typing import List, Dict, Any


class Experience(object):
    """docstring for Experience"""

    def __init__(self, vehs_prev: List[Vehicle], vehs: List[Vehicle], feasible_actions_all_vehs: List[List[Action]],
     time: float, num_requests: int, 
     SR, mean_SR = 0):
        super(Experience, self).__init__()
        self.vehs_prev = vehs_prev
        self.vehs = vehs
        self.feasible_actions_all_vehs = feasible_actions_all_vehs
        self.time = time
        self.num_requests = num_requests

        self.SR_flat = SR  #Either source or pair service rates, flattened
        self.mean_SR = mean_SR

        self.representation: Dict[bool, Any] = {}


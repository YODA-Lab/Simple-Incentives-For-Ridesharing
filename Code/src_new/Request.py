from typing import Optional
from math import floor


class Request(object):
    """
    A Request is the atomic unit in an Action.

    It represents a single customer's *request* for a ride
    """

    MAX_PICKUP_DELAY: float = 300.0
    MAX_DROPOFF_DELAY: float = 600.0
    MAX_PICKUP_PENALTY: float = 301.0
    MAX_DROPOFF_PENALTY: float = 601.0

    def __init__(self,
                 request_id: int,
                 source: int,
                 destination: int,
                 current_time: float,
                 travel_time: float,
                 base_price: float,
                 epoch_len: float,
                 GAMMA: float
                 ):
        self.request_id = request_id
        self.pickup = source
        self.dropoff = destination
        self.pickup_time = None
        self.dropoff_time = None
        self.base_price = base_price
        self.num_timesteps = travel_time / epoch_len
        self.gamma = GAMMA

        self.pickup_deadline = current_time + self.MAX_PICKUP_DELAY
        self.dropoff_deadline = current_time + travel_time + self.MAX_DROPOFF_DELAY
        self.price: Optional[float] = None

    def get_reward(self, price: float) -> float:
        """Function that discounts the price obtained over multiple time-steps"""
        reward_per_timestep = price / self.num_timesteps

        # For all but the last time_step
        last_step = floor(self.num_timesteps)
        discounted_reward = reward_per_timestep * (1 - self.gamma ** last_step) / (1 - self.gamma)

        # For the last time-step
        discounted_reward += (self.gamma ** last_step) * (price - reward_per_timestep * last_step)

        return discounted_reward

    def __deepcopy__(self, memo):
        return self

    def __str__(self):
        return("{}->{}".format(self.pickup, self.dropoff))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.request_id)

    def __eq__(self, other):
        # Request is only comparable with other Requests
        if isinstance(other, self.__class__):
            # If the ids are the same, they are equal
            if (self.request_id == other.request_id):
                return True

        return False

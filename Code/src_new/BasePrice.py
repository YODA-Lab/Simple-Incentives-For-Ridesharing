from abc import ABC, abstractmethod
from math import ceil


class BasePrice(ABC):
    """
    Defines an interface for mapping requests to the rewards associated with serving them
    """

    def __init__(self):
        super(BasePrice, self).__init__()

    @abstractmethod
    def get_base_price(self, distance: float, time: float) -> float:
        """Function that defines the reward to maximise"""
        raise NotImplementedError


class FlatPrice(BasePrice):
    """
    Paradigm in which serving any request has value $1.
    Maximising for this reward leads to a policy that maximises the number of requests served.
    """

    def __init__(self):
        super(FlatPrice, self).__init__()

    def get_base_price(self, distance: float, time: float) -> float:
        """The reward is 1 for every request"""
        return 1


class UberXPrice(BasePrice):
    """
    Paradigm in which serving a request gives you revenue equal to the 'price'
    Maximising for this reward leads to a policy that maximises revenue.
    For Manhattan, the pricing structure for Uber is as follows:
        Base Fare: $7.19, Per Minute (including pick-up): $0.67, Per Mile (including drop-off): $1.48
    """

    def __init__(self):
        super(UberXPrice, self).__init__()

    def get_base_price(self, distance: float, time: float) -> float:
        """The reward is defined as the price of an UberX ride taken in Manhattan"""
        time = ceil(time / 60)  # in minutes
        distance = ceil(distance)

        return 7.19 + 0.67 * time + 1.48 * distance


class UberPoolPrice(UberXPrice):
    """
    Paradigm in which serving a request gives you revenue equal to the 'price'
    Maximising for this reward leads to a policy that maximises revenue.
    We assume that the UberPoolPrice is some \\alpha fracton of the UberX price
    """

    def __init__(self, alpha: float):
        super(UberPoolPrice, self).__init__()
        self.alpha = alpha

    def get_base_price(self, distance: float, time: float) -> float:
        """The reward is defined as an \\alpha fraction of the price of an UberX ride taken in Manhattan"""
        return self.alpha * super(UberPoolPrice, self).get_base_price(distance, time)

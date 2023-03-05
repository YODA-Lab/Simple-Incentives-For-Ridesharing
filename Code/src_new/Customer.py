from Request import Request
from utils import IncorrectUsageError

from abc import ABC, abstractmethod
from random import uniform
from numpy.random import logistic
from math import exp, log


class Customer(ABC):
    """A class that models customer behaviour. Once a customer is sent a quote, they have to choose whether or not to accept it."""

    def __init__(self):
        super(Customer, self).__init__()

    @abstractmethod
    def sample_value(self, request: Request) -> float:
        raise NotImplementedError

    @abstractmethod
    def virtual_value(self, request: Request, value: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def pr_accept(self, request: Request, price: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_accept(self, request: Request, price: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def prob_to_price(self, request: Request, prob: float) -> float:
        raise NotImplementedError


class LogisticCustomer(Customer):
    """
    This class models a general class of behaviour models where the probability of a request being accepted at a given price 'x' can be given by the function sigm(ax/BP + b)
    where, a and b are constants for different value functions, BP is the Base Price and sigm(x) is the sigmoid or logistic function
    """

    def __init__(self, a: float, b: float) -> None:
        super(LogisticCustomer, self).__init__()

        self.a, self. b = a, b

    def sample_value(self, request: Request) -> float:
        # Instead of directly using the probability of accepting a request, we can also use this notion of intrinsic value to determine whether or not a customer accepts a given request
        # The probability of accepting a request can be seen as the 1 - CDF of the 'intrinsic value' of the customer
        # Here, we define the 'intrinsic value' as the maximum amount a customer is willing to pay for a given request.

        # We sample from this distribution of intrinsic value using numpy
        # Convert the CDF from 1 - sigm(ax/BP + b) to the form that is expected by numpy function sigm((x - loc)/scale)
        #   here, we use the property 1 - sigm(ax/BP + b) = sigm(-(ax/BP + b))

        # Then, scale--the width of the distribution is:
        scale = - request.base_price / self.a

        # And loc--The center of the logistic distribution is:
        loc = self.b * scale

        # Sample using numpy function and return
        return logistic(loc=loc, scale=scale)

    def virtual_value(self, request: Request, value: float) -> float:
        # Get the 'virtual value' corresponding to a given (request, value) pair
        # Virtual value \phi for a given pdf f, cdf F and value v is given by:
        #   \phi(v) = v - (1 - F(v))/f(v)
        F = 1 - self.pr_accept(request, value)
        f = - F * (1 - F) * self.a / request.base_price  # from the fact that sigm'(x) = sigm(x) * (1 - sigm(x))

        phi = value - (1 - F) / f

        return phi

    def pr_accept(self, request: Request, price: float) -> float:
        return 1 / (1 + exp(-(self.a * price / request.base_price + self.b)))

    def is_accept(self, request: Request, price: float) -> bool:
        # We probabilistically accept or decline requests according to the customer model

        y = uniform(0, 1)
        return y <= self.pr_accept(request, price)

    def prob_to_price(self, request: Request, prob: float) -> float:
        # Maps a given 'probability of accepting' to its corresponding price
        #   If prob = 1/(1 + e^(-(ax + b))), x = 1/a * (log(prob / (1 - prob)) - b)
        assert 0 <= prob <= 1  # checking bounds for probability variable

        # Making sure probability isn't exactly 0 or 1 to prevent numerical issues
        prob = min(max(1e-6, prob), 1 - 1e-6)

        # Calculating price according to the formula
        price = request.base_price / self.a * (log(prob / (1 - prob)) - self.b)

        return price


#   Specific model for Customer behaviour published by Uber in [Dynamic Pricing and Matching in Ride-Hailing Platforms](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3258234)
class UberCustomer(LogisticCustomer):
    """The probability of accepting a request at price x in this model can be given by the function sigm(-0.67x/BP + 1.69)"""

    def __init__(self):
        super(UberCustomer, self).__init__(a=-0.67, b=1.69)


#   Modification of Uber Customer behaviour
#   Probability of acceptance at BasePrice is same as in Uber, but the probability of acceptance at 3 * BasePrice is fixed to 0.1
#   A new function is plotted by interpolating these points
class PriceConsciousCustomer(LogisticCustomer):
    """The probability of accepting a request at price x in this model can be given by the function sigm(-1.609x/BP + 2.629)"""

    def __init__(self):
        super(PriceConsciousCustomer, self).__init__(a=-1.609, b=2.629)


#   Customer that always accepts at any price
#   Intended use: Backwards compatibility with https://github.com/sanketkshah/NeurADP-for-Ride-Pooling
#   IMPORTANT: YOU ONLY USE THIS WITH BasePricePricer
class AlwaysAcceptCustomer(Customer):
    """Customer that always accepts the request, regardless of the price."""

    def __init__(self):
        super(AlwaysAcceptCustomer, self).__init__()

    def sample_value(self, request: Request) -> float:
        raise IncorrectUsageError

    def virtual_value(self, request: Request, value: float) -> float:
        raise IncorrectUsageError

    def pr_accept(self, request: Request, price: float) -> float:
        return 1

    def is_accept(self, request: Request, price: float) -> bool:
        return True

    def prob_to_price(self, request: Request, prob: float) -> float:
        raise IncorrectUsageError

from Customer import Customer
from Request import Request
from Action import Action
from Environment import Environment
from CentralAgent import CentralAgent

from typing import List, Tuple, Dict, Set

from abc import ABC, abstractmethod
from scipy.optimize import minimize, linear_sum_assignment
from numpy import zeros


class Pricer(ABC):
    """
    Defines an interface for pricing requests.
    Currently requires the probability of acceptance to be some kind of sigmoid function of the base_price
    """

    def __init__(self):
        super(Pricer, self).__init__()

    @abstractmethod
    def get_prices(
        self,
        requests: List[Request],
        scored_actions_all_vehs: List[List[Tuple[Action, float]]],
        customer: Customer
    ) -> Dict[Request, float]:

        # Function sets the price field for each request
        raise NotImplementedError


class BasePricePricer(Pricer):
    """A pricer that charges the BasePrice for every request."""

    def __init__(self):
        super(BasePricePricer, self).__init__()

    def get_prices(
        self,
        requests: List[Request],
        scored_actions_all_vehs: List[List[Tuple[Action, float]]],
        customer: Customer
    ) -> Dict[Request, float]:

        return {request: request.base_price for request in requests}


# TODO(?): Optimise NaivePricer so that optimisation is run only once
class NaivePricer(Pricer):
    """
    Sets the best static price for 1 request and 1 vehicle as the default price
    The best static price x is the price that maximises the expected revenue
    This is given by argmax_{x} x * sigm(ax/BP + b)

    Note: This ignores the ratio of requests to vehicles and the feasibility graph
    """

    def __init__(self, customer: Customer):
        super(NaivePricer, self).__init__()
        self.customer = customer

    def get_prices(
        self,
        requests: List[Request],
        scored_actions_all_vehs: List[List[Tuple[Action, float]]],
        customer: Customer
    ) -> Dict[Request, float]:

        req_to_price = {}
        for request in requests:
            # Define Expected Revenue
            def expected_revenue(x):
                pr_x = customer.pr_accept(request, x)
                expected_revenue = x * pr_x
                return expected_revenue

            # Maximise expected revenue using scipy interface
            optima = minimize(lambda x: - expected_revenue(x[0]), x0=(request.base_price))  # uses numerical minimisation; the negative sign is because we're minimising, instead of maximising
            price = optima.x[0]

            # Set price field
            req_to_price[request] = price

        return req_to_price


class DecompositionPricer(Pricer):
    """
    This pricing aproach tries to decompose the overall request-vehicle matching graph into a separate matching graph for each vehicle
    It does this by iteratively matching requests to vehicles in rounds
      In each round you add up to 1 request to the matching graph for each vehicle
      Choosing which vehicle to match a request to is done using an LP
    """

    def __init__(self, customer: Customer, envt: Environment):
        super(DecompositionPricer, self).__init__()
        self.customer = customer
        assert envt.MAX_CAPACITY == 1  # this pricing only works in unit capacity cases

    def get_prices(
        self,
        requests: List[Request],
        scored_actions_all_vehs: List[List[Tuple[Action, float]]],
        customer: Customer
    ) -> Dict[Request, float]:

        # Helper function to determine the value of a given vehicle graph
        def get_max_expected_revenue(requests, vehicle):
            # Define Expected Revenue
            def get_expected_revenue(prices):
                # Sanity check
                assert len(prices) > 0

                # Sort prices in descending order
                sorted_prices = sorted(zip(requests, prices), reverse=True, key=lambda x: x[1])

                # Then, try to assign the copy with the highest price first
                expected_revenue = 0
                remaining_prob = 1
                for request, price in sorted_prices:
                    pr_accept = customer.pr_accept(request, price)
                    additional_revenue = remaining_prob * pr_accept * price

                    expected_revenue += additional_revenue
                    remaining_prob *= 1 - pr_accept

                return expected_revenue

            # Sanity check
            assert len(requests) >= 1

            # Maximise expected revenue using scipy interface
            optima = minimize(lambda x: - get_expected_revenue(x), x0=tuple(request.base_price for request in requests))  # uses numerical minimisation; the negative sign is because we're minimising, instead of maximising
            max_expected_revenue = - optima.fun
            prices = optima.x

            return max_expected_revenue, prices

        # PREPROCESS FEASIBILITY GRAPH
        vehicles = list(range(len(scored_actions_all_vehs)))

        # CREATE DECOMPOSED GRAPH
        current_decomposition: Dict[int, List[Request]] = {vehicle: [] for vehicle in vehicles}
        current_revenue = {vehicle: 0 for vehicle in vehicles}
        active_requests = [request for request in requests]

        # In each round
        while True:
            # Create weight matrix for matching
            match_utility = zeros((len(active_requests), len(vehicles)))  # 1e-6 is an arbitrary small number

            # Enter the values for this matrix
            # For each active request
            for request_idx, request in enumerate(active_requests):
                # Determine value of adding an active request to a vehicle graph
                # For each vehicle
                for vehicle_idx, vehicle in enumerate(vehicles):
                    # If the request can be matched to it
                    # TODO: Change. Hacky; only works for unit capacity.
                    for action, score in scored_actions_all_vehs[vehicle]:
                        if request in action.requests:
                            # Determine value of matching
                            # TODO: Incorporate future value. Current only considers price.
                            potential_decomposition = [request, *current_decomposition[vehicle]]
                            new_revenue, _ = get_max_expected_revenue(potential_decomposition, vehicle)

                            # Add it to the weight matrix
                            match_utility[request_idx][vehicle_idx] = new_revenue - current_revenue[vehicle]

            # Create the best weighted match of requests to vehicle graphs
            row_index, col_index = linear_sum_assignment(- match_utility)

            # Update variables
            matched_reqs = []
            for request_idx, vehicle_idx in zip(row_index, col_index):
                if (match_utility[request_idx, vehicle_idx] <= 0):
                    continue

                request = active_requests[request_idx]
                matched_vehicle = vehicles[vehicle_idx]

                matched_reqs.append(request)
                current_decomposition[matched_vehicle].append(request)
                current_revenue[matched_vehicle] += match_utility[request_idx, vehicle_idx]

            # If no progress has been made, exit
            # Else, proceed to next round
            new_active_requests = [request for request in active_requests if request not in matched_reqs]
            if not new_active_requests or (set(new_active_requests) == set(active_requests)):
                break
            else:
                active_requests = new_active_requests

        # Determine optimal pricing for given decomposition
        req_to_price = {}
        total_revenue = 0
        for vehicle in vehicles:
            if current_decomposition[vehicle]:
                vehicle_revenue, prices = get_max_expected_revenue(current_decomposition[vehicle], vehicle)

                total_revenue += vehicle_revenue
                for idx, request in enumerate(current_decomposition[vehicle]):
                    req_to_price[request] = prices[idx]

        print(f"Expected Revenue (Train): {total_revenue}")

        return req_to_price


class MyersonPricer(Pricer):
    """
    This pricing approach calculates the price in 2 steps:
        (1) Calculate the probability that a request would be assigned in the expectation
        (2) Use this probability to calculate the price

    To do Step 1, we sample possible internal valuations for different customers and then match
    according to those valuations. Then, to calculate the probability of assignment, we see how often
    the requests are matched across these different samples.

    In Step 2, we have to use these probabilities to calculate a price. The way we do this is that
    we choose a price such that, at that price, the probability of assignment of the request is the
    same as the probability of assignment calculated in Step 1. This is along the lines of the
    agorithm suggested in ['Multi-parameter mechanism design and sequential posted pricing'](https://dl.acm.org/doi/abs/10.1145/1806689.1806733)
    """

    def __init__(
        self,
        customer: Customer,
        central_agent: CentralAgent,
        NUM_SAMPLES: int = 100,
        is_ignore_future: bool = False
    ):
        super(MyersonPricer, self).__init__()
        self.customer = customer
        self.central_agent = central_agent
        self.NUM_SAMPLES = NUM_SAMPLES
        self.is_ignore_future = is_ignore_future

    def get_prices(
        self,
        requests: List[Request],
        scored_actions_all_vehs: List[List[Tuple[Action, float]]],
        customer: Customer
    ) -> Dict[Request, float]:

        return self._get_prices_naive(requests, scored_actions_all_vehs, customer)

    def _get_prices_naive(
        self,
        requests: List[Request],
        scored_actions_all_vehs: List[List[Tuple[Action, float]]],
        customer: Customer
    ):
        match_freq = {request: 0 for request in requests}
        for _ in range(self.NUM_SAMPLES):
            # Get a sample of intrinsic values for customers
            value = {request: self.customer.sample_value(request)
                     for request in requests}

            # Score actions according to 'utility'
            action_utilities_all_vehs = []
            for scored_actions in scored_actions_all_vehs:
                action_utilities = []
                for action, future_value in scored_actions:
                    reward = sum([request.get_reward(
                                    self.customer.virtual_value(request, value[request]))
                                  for request in action.requests])
                    if self.is_ignore_future:
                        future_value = 0
                    action_utilities.append((action, reward + future_value))
                action_utilities_all_vehs.append(action_utilities)

            # Find the optimal assignment of actions to vehicles
            final_actions = self.central_agent.choose_actions(action_utilities_all_vehs, is_training=False)

            # Update match frequency for requests in each optimal matching
            for action, _ in final_actions:
                for request in action.requests:
                    match_freq[request] += 1

        # Determine price according to probability of assignment
        req_to_price = {request: self.customer.prob_to_price(request, freq / self.NUM_SAMPLES)
                        for request, freq in match_freq.items()}

        return req_to_price

    def _get_prices_optimised(
        self,
        requests: List[Request],
        scored_actions_all_vehs: List[List[Tuple[Action, float]]],
        customer: Customer
    ):
        # Get samples of intrinsic value for customers
        values_all_samples = [{request: self.customer.sample_value(request)
                              for request in requests}
                              for _ in range(self.NUM_SAMPLES)]

        # Score actions according to 'utility'
        action_utilities_all_vehs = []
        for scored_actions in scored_actions_all_vehs:
            action_utilities = []
            for action, future_value in scored_actions:
                if self.is_ignore_future:
                    future_value = 0
                utilities = [(sum([request.get_reward(self.customer.virtual_value(request, values[request]))
                                   for request in action.requests]) + future_value)
                             for values in values_all_samples]
                action_utilities.append((action, utilities))
            action_utilities_all_vehs.append(action_utilities)

        # Find optimal assignment for each possible sample
        final_actions_all_samples = self.central_agent.choose_actions_multiutil(action_utilities_all_vehs, is_training=False)

        # Save list of requests in each optimal matching
        matched_reqs_all_samples: List[Set[Request]] = []
        for final_actions in final_actions_all_samples:
            matched_reqs = [request for action, _ in final_actions for request in action.requests]
            matched_reqs_all_samples.append(set(matched_reqs))

        # Calculate the probability of assignment across different samples
        match_freq = {request: sum([1
                                    for matched_reqs in matched_reqs_all_samples
                                    if request in matched_reqs])
                      for request in requests}

        # Determine price according to probability of assignment
        req_to_price = {request: self.customer.prob_to_price(request, freq / self.NUM_SAMPLES)
                        for request, freq in match_freq.items()}

        return req_to_price

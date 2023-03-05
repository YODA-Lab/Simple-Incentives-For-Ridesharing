from Vehicle import Vehicle
from Action import Action
from Request import Request
from Path import PathNode, RequestInfo
from BasePrice import BasePrice

from typing import List, Generator, Tuple, Deque, Dict

from abc import ABCMeta, abstractmethod
from random import choices
from pandas import read_csv
from collections import deque
from docplex.mp.model import Model  # type: ignore
import numpy as np
import re
import json
import os
from random import randint, random, shuffle

from collections import defaultdict as ddict
from copy import deepcopy

from utils import  add_driver_fairness_to_score, add_fairness_to_score ,gini, change_variance_driver, change_variance_rider, get_fscore
class Environment(metaclass=ABCMeta):
    """Defines a class for simulating the Environment for the RL agent"""

    REQUEST_HISTORY_SIZE: int = 1000

    def __init__(self, NUM_LOCATIONS: int, MAX_CAPACITY: int, EPOCH_LENGTH: int, NUM_VEHS: int, START_EPOCH: float, STOP_EPOCH: float, DATA_DIR: str, GAMMA: float, BASE_PRICE: BasePrice, VALUE_FUNCTION: str=''):
        # Load environment
        self.NUM_LOCATIONS = NUM_LOCATIONS
        self.MAX_CAPACITY = MAX_CAPACITY
        self.EPOCH_LENGTH = EPOCH_LENGTH
        self.NUM_VEHS = NUM_VEHS
        self.START_EPOCH = START_EPOCH
        self.STOP_EPOCH = STOP_EPOCH
        self.DATA_DIR = DATA_DIR
        self.GAMMA = GAMMA
        self.BASE_PRICE = BASE_PRICE

        self.num_days_trained = 0
        self.recent_request_history: Deque[Request] = deque(maxlen=self.REQUEST_HISTORY_SIZE)
        self.current_time: int = 0
        self.value_function = VALUE_FUNCTION

        self.carryover_reqs = []



    @abstractmethod
    def initialise_environment(self):
        raise NotImplementedError

    # @abstractmethod
    # def generate_feasible_assignments(self, feasible_actions_all_vehs):
    #     raise NotImplementedError

    # @abstractmethod
    # def get_reward_completed(self):
    #     raise NotImplementedError

    @abstractmethod
    def get_request_batch(self, day):
        raise NotImplementedError

    @abstractmethod
    def get_travel_time(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_next_location(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_initial_states(self, num_vehs, is_training):
        raise NotImplementedError

    @abstractmethod
    def has_valid_path(self, veh: Vehicle) -> bool:
        raise NotImplementedError

    def simulate_motion(self, vehs: List[Vehicle], current_requests: List[Request] = [], rebalance: bool = True) -> None:
        # Move all vehs
        vehs_to_rebalance: Dict[Vehicle, float] = {}
        for veh in vehs:
            time_remaining: float = self.EPOCH_LENGTH
            time_remaining = self._move_veh(veh, time_remaining)
            # If it has visited all the locations it needs to and has time left, rebalance
            if (time_remaining > 0):
                vehs_to_rebalance[veh] = time_remaining

        # Update recent_requests list
        self.update_recent_requests(current_requests)

        # Perform Rebalancing
        if (rebalance and vehs_to_rebalance):
            rebalancing_targets = self._get_rebalance_targets(list(vehs_to_rebalance.keys()))

            # Move vehicles according to the rebalancing_targets
            for veh, target in rebalancing_targets.items():
                time_remaining = vehs_to_rebalance[veh]

                # Insert dummy target
                veh.path.requests.append(RequestInfo(target, False, True))
                veh.path.request_order.append(PathNode(False, 0))  # adds pickup location to 'to-visit' list

                # Move according to dummy target
                self._move_veh(veh, time_remaining)

                # Remove dummy target
                veh.path.request_order.clear()
                veh.path.requests.clear()
                veh.path.current_capacity = 0
                veh.path.total_delay = 0
                veh.path.response_delay = 0

    def _move_veh(self, veh: Vehicle, time_remaining: float) -> float:
        while(time_remaining >= 0):
            time_remaining -= veh.position.time_to_next_location

            # If we reach an intersection, make a decision about where to go next
            if (time_remaining >= 0):
                # If the intersection is an existing pick-up or drop-off location, update the Vehicle's path
                if (veh.position.next_location == veh.path.get_next_location()):
                    next_node = veh.path.request_order[0]
                    relevant_request = veh.path.requests[next_node.relevant_request_id].request
                    # print(relevant_request.pickup_time)
                    if next_node.is_dropoff:
                        relevant_request.dropoff_time = self.current_time
                    else:
                        relevant_request.pickup_time = self.current_time

                    veh.path.visit_next_location(self.current_time + self.EPOCH_LENGTH - time_remaining)

                # Go to the next location in the path, if it exists
                if (not veh.path.is_empty()):
                    next_location = self.get_next_location(veh.position.next_location, veh.path.get_next_location())
                    veh.position.time_to_next_location = self.get_travel_time(veh.position.next_location, next_location)
                    veh.position.next_location = next_location

                # If no additional locations need to be visited, stop
                else:
                    veh.position.time_to_next_location = 0
                    break
            # Else, continue down the road you're on
            else:
                veh.position.time_to_next_location -= (time_remaining + veh.position.time_to_next_location)

        return time_remaining

    # TODO: Use Scipy & numpy to do it more efficiently
    def _get_rebalance_targets(self, vehs: List[Vehicle]) -> Dict[Vehicle, Request]:
        # Get a list of possible targets by sampling from recent_requests
        possible_targets: List[Request] = choices(self.recent_request_history, k=len(vehs))

        # Solve an LP to assign each veh to closest possible target
        model = Model()

        # Define variables, a matrix defining the assignment of vehs to targets
        assignments = model.continuous_var_matrix(range(len(vehs)), range(len(possible_targets)), name='assignments')

        # Make sure one veh can only be assigned to one target
        for veh_id in range(len(vehs)):
            model.add_constraint(model.sum(assignments[veh_id, target_id] for target_id in range(len(possible_targets))) == 1)

        # Make sure one target can only be assigned to 1
        for target_id in range(len(possible_targets)):
            model.add_constraint(model.sum(assignments[veh_id, target_id] for veh_id in range(len(vehs))) == 1)

        # Define the objective: Minimise distance travelled
        model.minimize(model.sum(assignments[veh_id, target_id] * self.get_travel_time(vehs[veh_id].position.next_location, possible_targets[target_id].pickup) for target_id in range(len(possible_targets)) for veh_id in range(len(vehs))))

        # Solve
        solution = model.solve()
        assert solution  # making sure that the model doesn't fail

        # Get the assigned targets
        assigned_targets: Dict[Vehicle, Request] = {}
        for veh_id in range(len(vehs)):
            for target_id in range(len(possible_targets)):
                if (solution.get_value(assignments[veh_id, target_id]) == 1):
                    assigned_targets[vehs[veh_id]] = possible_targets[target_id]

        # Ensure that there's a target for every vehicle
        for veh in vehs:
            assert veh in assigned_targets

        return assigned_targets

    def get_reward(self, action: Action) -> float:
        """
        Return the reward to a vehicle for a given (feasible) action.

        (Feasibility is not checked!)
        Defined in Environment class because of Reinforcement Learning
        convention in literature.
        """

        # The total reward is the sum of rewards of all requests in a given action
        total_reward = 0.0

        for request in action.requests:
            # Sanity Check: Ensure that the requests associated with the actions have been priced
            assert request.price is not None

            total_reward += request.get_reward(request.price)

        return total_reward

    def get_revenue(self, action: Action) -> float:
        """
        Return the revenue obtained for a given (feasible) action.
        """

        # The total revenue is the sum of prices of all requests in a given action
        revenue = 0.0
        for request in action.requests:
            # Sanity Check: Ensure that the requests associated with the actions have been priced
            assert request.price is not None

            revenue += request.price

        return revenue

    def update_recent_requests(self, recent_requests: List[Request]):
        self.recent_request_history.extend(recent_requests)




class MapEnvironment(Environment):
    """
    Environment using either ny or conworld that can support gini updates, delay based reward calculation, and carryover
    """

    NUM_MAX_VEHS: int = 3000
    NUM_LOCATIONS: int = 4461
    # NUM_LOCATIONS: int = 205

    def __init__(
        self,
        NUM_VEHS: int,
        START_EPOCH: float,
        STOP_EPOCH: float,
        MAX_CAPACITY,
        GAMMA: float,
        BASE_PRICE: BasePrice,
        DATA_DIR: str = '../data/conworld/',
        EPOCH_LENGTH: int = 60,
        VALUE_FUNCTION: str = '',
        DELAY_TYPE: str = 'dropoff_delay', #options: pickup_delay, dropoff_delay
        START_DAY: int = 11,
        GINI_DISCOUNT = 1,
        filtered=False,
        ALPHA = 0.,
        BETA: float = 1.,
        FAIRNESS_TARGET = '',
        FAIRTYPE = 'pair', #options : pair, src,
        ALPHA_D = 0.,
        DELTA = 0. #Driver fairness weight
    ):
        #if DATA_DIR == '../data/ny/':
        #    self.NUM_LOCATIONS: int = 4461

        super().__init__(
            NUM_LOCATIONS=self.NUM_LOCATIONS,
            MAX_CAPACITY=MAX_CAPACITY,
            EPOCH_LENGTH=EPOCH_LENGTH,
            NUM_VEHS=NUM_VEHS,
            START_EPOCH=START_EPOCH,
            STOP_EPOCH=STOP_EPOCH,
            DATA_DIR=DATA_DIR,
            GAMMA=GAMMA,
            BASE_PRICE=BASE_PRICE,
            VALUE_FUNCTION=VALUE_FUNCTION
        )
        self.filtered = filtered
        self.initialise_environment()
        
        #For inter-zone tracking
        ZONE_MAPPING_FILE: str = os.path.join(self.DATA_DIR, 'neighbourhood_zones.json')
        self.zone_mapping_data = json.load(open(ZONE_MAPPING_FILE, 'r'))
        if self.DATA_DIR == '../data/conworld/':
            self.zone_mapping = {pt["id"]: pt["zone"] for pt in self.zone_mapping_data}    
        else:
            self.zone_mapping = {pt["id"]: pt["neighbourhood"] for pt in self.zone_mapping_data}
        
        self.sort_by = 'neighbourhood'
        b = json.load(open(os.path.join(self.DATA_DIR, "neigh_zipcodes.json"), 'r'))
        neigh_centroids = json.load(open(os.path.join(self.DATA_DIR, "neighbourhood_centroids.json"), 'r'))
        neigh_list = sorted(b.keys(), key=lambda x: neigh_centroids[x][1]) # sort by longitude
        zip_list = []
        for v in b.values():
            zip_list.extend(v)
        self.keys_list = neigh_list if self.sort_by == 'neighbourhood' else zip_list
        
        #For service rate
        # For calculating fairness reward
        self.fairtype = FAIRTYPE
        self.fairness_target = FAIRNESS_TARGET  #Fair vehicles/Fair Requests/ Positive score requests
        self.beta = BETA  #WWeight of the fairness term
        self.alpha = ALPHA
        self.service_rate = 0
        self.accepted_requests = 0
        self.total_requests = 0
        self.min_pair_sr = 0
        self.mean_pair_sr = 0
        self.pair_sr = {}  #Service rates for zone pairs
        self.source_sr = {} #service rates by origin zones 
        for item in self.keys_list:
            self.pair_sr[item] = {}
            self.source_sr[item] = [0,0,0]
            for item2 in self.keys_list:
                self.pair_sr[item][item2] = [0,0,0]   #[from][to]: [pair_sr,total served, total seen]
        
        self.pair_sr_overall = deepcopy(self.pair_sr)  #Actual service rates for zone pairs (Not discounted)
        self.source_sr_overall = deepcopy(self.source_sr)
        
        #For driver fairness
        self.delta = DELTA #Weight of the driver fairness term
        self.alpha_d = ALPHA_D
        self.driver_earnings = [0 for v in range(NUM_VEHS)]
        self.norm_disc_driver_earnings = [0 for v in range(NUM_VEHS)]
        self.driver_avg = 0
        self.driver_gini = 0
        self.driver_min = 0


    def initialise_environment(self):
        print('Loading Environment...')

        TRAVELTIME_FILE: str = self.DATA_DIR + 'zone_traveltime_short.csv'
        self.travel_times = read_csv(TRAVELTIME_FILE, header=None).values

        DIST_FILE: str = self.DATA_DIR + 'zone_dist_short.csv'
        self.distances = read_csv(DIST_FILE, header=None).values

        SHORTESTPATH_FILE: str = self.DATA_DIR + 'zone_path.csv'
        self.shortest_paths = read_csv(SHORTESTPATH_FILE, header=None).values

        IGNOREDZONES_FILE: str = self.DATA_DIR + 'ignorezonelist.txt'
        self.ignored_zones = read_csv(IGNOREDZONES_FILE, header=None).values.flatten()

        INITIALZONES_FILE: str = self.DATA_DIR + 'taxi_3000_final.txt'
        self.initial_zones = read_csv(INITIALZONES_FILE, header=None).values.flatten()

        assert (self.EPOCH_LENGTH == 60) or (self.EPOCH_LENGTH == 30) or (self.EPOCH_LENGTH == 10)
        if self.filtered:
            self.DATA_FILE_PREFIX: str = "{}files_{}sec/Further_Filtered/filtered_test_flow_5000_".format(self.DATA_DIR, int(self.EPOCH_LENGTH))
        else:
            self.DATA_FILE_PREFIX: str = "{}files_{}sec/test_flow_5000_".format(self.DATA_DIR, int(self.EPOCH_LENGTH))

    def get_request_batch(self,
                            day: int = 2,
                            downsample: float = 1) -> Generator[List[Request], None, None]:

        assert 0 < downsample <= 1
        request_id = 0

        def is_in_time_range(current_time):
            current_hour = int(current_time / 3600)
            return True if (current_hour >= self.START_EPOCH / 3600 and current_hour < self.STOP_EPOCH / 3600) else False

        # Open file to read
        with open(self.DATA_FILE_PREFIX + str(day) + '.txt', 'r') as data_file:
            num_batches: int = int(data_file.readline().strip())

            # Defines the 2 possible RE for lines in the data file
            new_epoch_re = re.compile(r'Flows:(\d+)-\d+')
            request_re = re.compile(r'(\d+),(\d+),(\d+)\.0')

            # Parsing rest of the file
            request_list: List[Request] = []
            is_first_epoch = True
            for line in data_file.readlines():
                line = line.strip()

                is_new_epoch = re.match(new_epoch_re, line)
                if (is_new_epoch is not None):
                    if not is_first_epoch:
                        if is_in_time_range(self.current_time):
                            yield request_list
                        request_list.clear()  # starting afresh for new batch
                    else:
                        is_first_epoch = False

                    current_epoch = int(is_new_epoch.group(1))
                    self.current_time = current_epoch * self.EPOCH_LENGTH
                else:
                    request_data = re.match(request_re, line)
                    assert request_data is not None  # Make sure there's nothing funky going on with the formatting

                    num_requests = int(request_data.group(3))
                    for _ in range(num_requests):
                        # Take request according to downsampled rate
                        rand_num = random()
                        if (rand_num > downsample):
                            continue

                        source = int(request_data.group(1))
                        destination = int(request_data.group(2))
                        if (source in self.ignored_zones or destination in self.ignored_zones or source == destination):
                            continue  # Filter problematic requests out

                        request_list.append(
                            Request(
                                request_id,
                                source,
                                destination,
                                self.current_time,
                                self.get_travel_time(source, destination),
                                self.get_request_value(source, destination),
                                self.EPOCH_LENGTH,
                                self.GAMMA
                            )
                        )
                        request_id += 1

            if is_in_time_range(self.current_time):
                yield request_list

    def get_travel_time(self, source: int, destination: int) -> float:
        return self.travel_times[source, destination]

    def get_dist(self, source: int, destination: int) -> float:
        return self.distances[source, destination]

    def get_next_location(self, source: int, destination: int) -> int:
        return self.shortest_paths[source, destination]

    def get_request_value(self, source, destination):
        travel_time = self.get_travel_time(source, destination)
        distance = self.get_dist(source, destination)

        base_price = self.BASE_PRICE.get_base_price(distance, travel_time)
        return base_price

    def get_initial_states(self, num_vehs: int, is_training: bool) -> List[int]:
        """Give initial states for num_vehs vehicles"""
        if (num_vehs > self.NUM_MAX_VEHS):
            print('Too many vehicles. Starting with random states.')
            is_training = True

        # If it's training, get random states
        if is_training:
            initial_states = []

            for _ in range(num_vehs):
                initial_state = randint(0, self.NUM_LOCATIONS - 1)
                # Make sure it's not an ignored zone
                while (initial_state in self.ignored_zones):
                    initial_state = randint(0, self.NUM_LOCATIONS - 1)

                initial_states.append(initial_state)
        # Else, pick deterministic initial states
        else :
            initial_states = self.initial_zones[:num_vehs]

        return initial_states

    def has_valid_path(self, veh: Vehicle) -> bool:
        """Attempt to check if the request order meets deadline and capacity constraints"""
        def invalid_path_trace(issue: str) -> bool:
            print(issue)
            print('Vehicle {}:'.format(veh.id))
            print('Requests -> {}'.format(veh.path.requests))
            print('Request Order -> {}'.format(veh.path.request_order))
            print()
            return False

        # Make sure that its current capacity is sensible
        if (veh.path.current_capacity < 0 or veh.path.current_capacity > self.MAX_CAPACITY):
            return invalid_path_trace('Invalid current capacity')

        # Make sure that it visits all the requests that it has accepted
        if (not veh.path.is_complete()):
            return invalid_path_trace('Incomplete path.')

        # Start at global_time and current_capacity
        current_time = self.current_time + veh.position.time_to_next_location
        current_location = veh.position.next_location
        current_capacity = veh.path.current_capacity

        # Iterate over path
        available_delay: float = 0
        response_delay: float = 0.
        for node_idx, node in enumerate(veh.path.request_order):
            next_location, deadline = veh.path.get_info(node)

            # Delay related checks
            travel_time = self.get_travel_time(current_location, next_location)
            if (current_time + travel_time > deadline):
                return invalid_path_trace('Does not meet deadline at node {}'.format(node_idx))

            current_time += travel_time
            current_location = next_location

            # Updating available delay
            if (node.expected_visit_time != current_time):
                invalid_path_trace("(Ignored) Visit time incorrect at node {}".format(node_idx))
                node.expected_visit_time = current_time

            if (node.is_dropoff):
                available_delay += deadline - node.expected_visit_time
            else:
                response_delay += deadline - node.expected_visit_time

            # Capacity related checks
            if (current_capacity > self.MAX_CAPACITY):
                return invalid_path_trace('Exceeds MAX_CAPACITY at node {}'.format(node_idx))

            if (node.is_dropoff):
                next_capacity = current_capacity - 1
            else:
                next_capacity = current_capacity + 1
            if (node.current_capacity != next_capacity):
                invalid_path_trace("(Ignored) Capacity incorrect at node {}".format(node_idx))
                node.current_capacity = next_capacity
            current_capacity = node.current_capacity

        # Check total_delay
        if (veh.path.total_delay != available_delay):
            invalid_path_trace("(Ignored) Total delay incorrect.")
        veh.path.total_delay = available_delay

        # Check response_delay
        veh.path.response_delay = response_delay

        return True

    """
    Gini calculations
    """
    def reset(self, flush=False):
        #Carrying over a fraction of the volume to the next day
        discount=0.33
        if flush:
            discount=0
        #Hard reset accepted requests and total requests always.
        self.accepted_requests*=0
        self.total_requests*=0
        #Keep some history of group data, as this is discounted per step
        for item in self.keys_list:
            _, seen, tot = self.source_sr[item] 
            self.source_sr[item] = [0,seen*discount,tot*discount]
            self.source_sr_overall[item] = [0,0,0]
            for item2 in self.keys_list:
                _, seen, tot = self.pair_sr[item][item2]
                self.pair_sr[item][item2] = [0,seen*discount,tot*discount]
                self.pair_sr_overall[item][item2] = [0,0,0]
        
        self.driver_earnings = [0 for v in range(self.NUM_VEHS)]

    def update_all(self, scored_final_actions, vehs, day =11, dropped_requests = []):
        """
        Update service rate and fairness metrics.
        """

        reqs = []
        for action, _ in scored_final_actions:
            reqs.extend(self.get_request_infos(action))

        #for dropped requests:
        dummy_action = Action(dropped_requests, None)

        reqs.extend(self.get_request_infos(dummy_action, dropped=True))

        #Update the service rate numbers
        #Only works without carryover

        #Discount the past
        sr_hist_discount = 0.999
        # sr_hist_discount = 1
        for k,g in self.pair_sr.items():
            for k2, [_,seen,total] in g.items():
                self.pair_sr[k][k2][1] = seen*sr_hist_discount
                self.pair_sr[k][k2][2] = total*sr_hist_discount
        for k, [_,seen,total] in self.source_sr.items():
            self.source_sr[k][1] = seen*sr_hist_discount
            self.source_sr[k][2] = total*sr_hist_discount
        
        #include present request info
        for req in reqs:
            self.total_requests +=1
            pickup = self.zone_mapping[req['pickup']]
            dropoff = self.zone_mapping[req['dropoff']]
            self.pair_sr[pickup][dropoff][2]+=1
            self.source_sr[pickup][2]+=1
            #also update overall sr
            self.pair_sr_overall[pickup][dropoff][2]+=1
            self.source_sr_overall[pickup][2]+=1
            if not req['dropped']:
                self.accepted_requests+=1
                self.source_sr[pickup][1]+=1
                self.pair_sr[pickup][dropoff][1]+=1

                self.source_sr_overall[pickup][1]+=1
                self.pair_sr_overall[pickup][dropoff][1]+=1
        
        #If any requests are served, update the service rates
        if len(reqs):
            self.service_rate = self.accepted_requests/self.total_requests
            pair_srs = []
            source_srs = []
            for k,g in self.pair_sr.items():
                for k2, [_,seen,total] in g.items():
                    if total:
                        pair_sr = seen/total
                        self.pair_sr[k][k2][0] = pair_sr
                        pair_srs.append(pair_sr)
            for k, [_,seen,total] in self.source_sr.items():
                if total:
                    self.source_sr[k][0] = seen/total
                    source_srs.append(seen/total)
            #update overall rates
            for k,g in self.pair_sr_overall.items():
                for k2, [_,seen,total] in g.items():
                    if total:
                        pair_sr_overall = seen/total
                        self.pair_sr_overall[k][k2][0] = pair_sr_overall
            for k, [_,seen,total] in self.source_sr_overall.items():
                if total:
                    self.source_sr_overall[k][0] = seen/total
            
            self.mean_pair_sr = np.mean(pair_srs)
            self.mean_source_sr = np.mean(source_srs)
            self.min_pair_sr = np.min(pair_srs)
            self.max_pair_sr = np.max(pair_srs)
            self.min_source_sr = np.min(source_srs)
            self.max_source_sr = np.max(source_srs)
            self.pair_sr_std = np.std(pair_srs)
            self.gini_pair_sr = gini(pair_srs)
            self.gini_source_sr = gini(source_srs)
        
        #For vehicles
        self.driver_earnings = [v.earning for v in vehs]
        self.driver_gini, self.driver_avg = gini(self.driver_earnings, return_mean=True)
        self.driver_min = np.min(self.driver_earnings)
        disc_earnings = np.array([v.discounted_value for v in vehs])
        self.norm_disc_driver_earnings = disc_earnings/max(1, np.max(disc_earnings))



    def get_request_infos(self, action, dropped=False):
        """
        Compute important statistics for requests in an action and return a list of request information dictionaries
        """
        request_info = []
        for request in action.requests:
            if not dropped:
                relevant_request_id = None
                for req_id, req_info in enumerate(action.new_path.requests):
                    if req_info.request==request:
                        relevant_request_id = req_id
                nodes = [node for node in action.new_path.request_order if node.relevant_request_id==relevant_request_id]

                pickup_delay = request.MAX_PICKUP_DELAY - (request.pickup_deadline - nodes[0].expected_visit_time)
                dropoff_delay = request.MAX_DROPOFF_DELAY - (request.dropoff_deadline - nodes[1].expected_visit_time)
                commute_delay = dropoff_delay - pickup_delay
            else:
                pickup_delay = request.MAX_PICKUP_PENALTY
                dropoff_delay = request.MAX_DROPOFF_PENALTY
                commute_delay = dropoff_delay

            req = {'pickup': request.pickup, 'dropoff':request.dropoff,
                   'pickup_delay':pickup_delay, 'dropoff_delay':dropoff_delay,
                   'commute_delay': commute_delay, 'request':request,
                   'dropped' : dropped}
            request_info.append(req)
        return request_info


    def get_dropped_requests(self, current_requests, scored_final_actions, time=None, carryover=False):
        """
        Return the requests that were not accepted in the current epoch
        If carrying over requests, drop only requests that cannot be fulfilled
        return the requests that can still be fulfilled as carryover_reqs

        If using in experience replay, cannot use current time.
        """
        if time == None:
            time = self.current_time

        accepted_reqs = []
        for action, _ in scored_final_actions:
            accepted_reqs.extend(list(action.requests))
        unserved_reqs = list(current_requests)
        for req in accepted_reqs:
            unserved_reqs.remove(req)

        carryover_reqs = []
        dropped_reqs = []
        if carryover == True:
            for request in unserved_reqs:
                if time<request.pickup_deadline:
                    carryover_reqs.append(request)
                else:
                    dropped_reqs.append(request)
        else:
            dropped_reqs = list(unserved_reqs) #Is this correct?

        return dropped_reqs, carryover_reqs

    def get_reward(self, action: Action) -> float:
        """
        Return the reward to a vehicle for a given (feasible) action.

        (Feasibility is not checked!)
        Defined in Environment class because of Reinforcement Learning
        convention in literature.
        """
        request_infos = self.get_request_infos(action)
        # The total reward is the sum of rewards of all requests in a given action

        profit_term = 0.0
        for req_info in request_infos:
            request = req_info['request']
            # Sanity Check: Ensure that the requests associated with the actions have been priced
            assert request.price is not None

            reward_term = request.get_reward(request.price)
            profit_term += reward_term

        #For FairNN (Raman et al. 2021)
        if self.value_function=='fairnn_rider':
            variance_term = change_variance_rider(self, action)
            total_reward = profit_term - self.lamda*variance_term
        elif self.value_function=='fairnn_driver':
            variance_term = change_variance_driver(self, action)
            total_reward = profit_term - self.lamda*variance_term
        else:
            total_reward = profit_term

        #adding driver and rider fairness

        #SIP
        assert self.alpha in [0,1]
        SIPass = 0
        if self.beta!=0:
            for req in action.requests:
                SIp = get_fscore(self,req)
                if self.alpha==0: #If considering SIP(+)
                    SIp = max(SIp,0)
                SIPass += SIp
        
        #SID
        assert self.alpha_d in [0,1]
        SIDrive = 0
        if self.delta!=0:
            dr_avg = np.mean(self.norm_disc_driver_earnings)
            dr_i = self.norm_disc_driver_earnings[action.veh_id]
            SId = self.delta*(dr_avg-dr_i)
            if self.alpha_d==0.0: #If considering SID(+)
                SId = max(SId,0)
            for req in action.requests:
                SIDrive += SId*req.base_price
        
        total_reward = total_reward + SIPass + SIDrive
        return total_reward

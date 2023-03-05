import heapq

from Request import Request
from Vehicle import Vehicle
from Action import Action
from Environment import Environment
from Oracle import Oracle

from typing import List, Dict, Tuple, Set, Any, Optional, Callable

from docplex.mp.model import Model  # type: ignore
from docplex.mp.linear import Var  # type: ignore
from random import shuffle, randint, random, uniform, choice
from copy import deepcopy

import numpy as np

class CentralAgent(object):
    """
    A CentralAgent arbitrates between different Vehicles.

    It takes the users 'preferences' for different actions as input
    and chooses the combination that maximises the sum of utilities
    for all the vehicles.

    It also trains the Vehicles' shared value function by
    querying the rewards from the environment and the next state.

    TODO: Document generalisation for multiple utils, each function
    """

    def __init__(
        self,
        envt: Environment,
        is_epsilon_greedy: bool = False
    ):
        super(CentralAgent, self).__init__()
        self.envt = envt
        self._choose = self._epsilon_greedy if is_epsilon_greedy else self._additive_noise

    # Wrapper function when there's only util per action
    def choose_actions(
        self,
        veh_action_choices_one_util: List[List[Tuple[Action, float]]],
        is_training: bool = True,
        epoch_num: int = 1
    ) -> List[Tuple[Action, float]]:

        veh_action_choices = [[(action, [util])
                               for action, util in actions_per_veh]
                              for actions_per_veh in veh_action_choices_one_util]

        return self._choose(veh_action_choices, is_training, epoch_num)[0]

    def choose_actions_multiutil(
        self,
        veh_action_choices: List[List[Tuple[Action, List[float]]]],
        is_training: bool = True,
        epoch_num: int = 1
    ) -> List[List[Tuple[Action, float]]]:

        return self._choose(veh_action_choices, is_training, epoch_num)

    def _epsilon_greedy(
        self,
        veh_action_choices: List[List[Tuple[Action, List[float]]]],
        is_training: bool = True,
        epoch_num: int = 1
    ) -> List[List[Tuple[Action, float]]]:

        # Decide whether or not to take random action
        rand_num = random()
        EPSILON = 0.1

        if not is_training or (rand_num > EPSILON):
            final_actions_all_utils = self._choose_actions_ILP(veh_action_choices)
        else:
            final_actions_all_utils = self._choose_actions_random(veh_action_choices)

        return final_actions_all_utils

    def _additive_noise(
        self,
        veh_action_choices: List[List[Tuple[Action, List[float]]]],
        is_training: bool = True,
        epoch_num: int = 1
    ) -> List[List[Tuple[Action, float]]]:

        # Define noise function for exploration
        def get_noise(variable: Var) -> float:
            # scale = 1 + (40 if 'x0,' in variable.get_name() else 10) / ((epoch_num + 1) * self.envt.NUM_VEHS)
            scale = 1 + (4000 if 'x0,' in variable.get_name() else 1000) / ((epoch_num + 1) * self.envt.NUM_VEHS)
            return uniform(0, scale) if is_training else 0

        final_actions = self._choose_actions_ILP(veh_action_choices, get_noise=get_noise)

        return final_actions

    def _choose_actions_ILP(
        self,
        veh_action_choices: List[List[Tuple[Action, List[float]]]],
        get_noise: Callable[[Var], float] = lambda x: 0
    ) -> List[List[Tuple[Action, float]]]:

        # Model as ILP
        model = Model()

        # For converting Action -> action_id and back
        action_to_id: Dict[Action, int] = {}
        id_to_action: Dict[int, Action] = {}
        current_action_id = 0

        # For constraint 2
        requests: Set[Request] = set()

        # Create decision variables and their coefficients in the objective
        # There is a decision variable for each (Action, Vehicle).
        # The coefficient is the util associated with the decision variable
        decision_variables: Dict[int, Dict[int, Tuple[Any, List[float]]]] = {}
        for veh_idx, scored_actions_per_veh in enumerate(veh_action_choices):
            for action, utils in scored_actions_per_veh:
                # Convert action -> id if it hasn't already been done
                if action not in action_to_id:
                    action_to_id[action] = current_action_id
                    id_to_action[current_action_id] = action
                    current_action_id += 1

                    action_id = current_action_id - 1
                    decision_variables[action_id] = {}
                else:
                    action_id = action_to_id[action]

                # Update set of requests in actions
                for request in action.requests:
                    if request not in requests:
                        requests.add(request)

                # Create variable for (action_id, veh_id)
                variable = model.binary_var(name='x{},{}'.format(action_id, veh_idx))

                # Save to decision_variable data structure
                decision_variables[action_id][veh_idx] = (variable, utils)

        # Create Constraint 1: Only one action per Vehicle
        for veh_idx in range(len(veh_action_choices)):
            veh_specific_variables: List[Any] = []
            for action_dict in decision_variables.values():
                if veh_idx in action_dict:
                    veh_specific_variables.append(action_dict[veh_idx])
            model.add_constraint(model.sum(variable for variable, _ in veh_specific_variables) == 1)

        # Create Constraint 2: Only one action per Request
        for request in requests:
            relevent_action_dicts: List[Dict[int, Tuple[Any, List[float]]]] = []
            for action_id in decision_variables:
                if (request in id_to_action[action_id].requests):
                    relevent_action_dicts.append(decision_variables[action_id])
            model.add_constraint(model.sum(variable for action_dict in relevent_action_dicts for variable, _ in action_dict.values()) <= 1)

        # Solve model for each set of utils
        #  Figure out how many sets of utils exist
        _, utils = veh_action_choices[0][0]
        NUM_UTILS = len(utils)

        #  For each set of utils, generate an allocation
        final_actions_all_utils: List[List[Tuple[Action, float]]] = []
        for util_idx in range(NUM_UTILS):
            # Create Objective
            score = model.sum((utils[util_idx] + get_noise(variable)) * variable
                              for action_dict in decision_variables.values()
                              for variable, utils in action_dict.values())
            model.maximize(score)

            # Solve ILP
            solution = model.solve()
            assert solution  # making sure that the model doesn't fail

            # Get vehicle specific actions from ILP solution
            assigned_actions: Dict[int, int] = {}
            for action_id, action_dict in decision_variables.items():
                for veh_idx, (variable, _) in action_dict.items():
                    if (solution.get_value(variable) == 1):
                        assigned_actions[veh_idx] = action_id

            # Use this to create list of final actions
            final_actions: List[Tuple[Action, float]] = []
            for veh_idx in range(len(veh_action_choices)):
                assigned_action_id = assigned_actions[veh_idx]
                assigned_action = id_to_action[assigned_action_id]
                scored_final_action: Optional[Tuple[Action, float]] = None
                for action, utils in veh_action_choices[veh_idx]:
                    if (action == assigned_action):
                        scored_final_action = (action, utils[util_idx])
                        break

                assert scored_final_action is not None
                final_actions.append(scored_final_action)

            # Save in variable
            final_actions_all_utils.append(final_actions)  # type: ignore

        return final_actions_all_utils

    def _choose_actions_random(
        self,
        veh_action_choices: List[List[Tuple[Action, List[float]]]]
    ) -> List[List[Tuple[Action, float]]]:

        final_actions: List[Optional[Tuple[Action, float]]] = [None] * len(veh_action_choices)
        consumed_requests: Set[Request] = set()

        # Create a random ordering of vehicles
        order = list(range(len(veh_action_choices)))
        shuffle(order)

        # Pick actions for each vehicle according to the order
        for veh_idx in order:
            # Create a list of feasible actions
            allowable_actions_idxs: List[int] = []
            for action_idx, (action, _) in enumerate(veh_action_choices[veh_idx]):
                is_consumed = [(request in consumed_requests) for request in action.requests]
                if sum(is_consumed) == 0:
                    allowable_actions_idxs.append(action_idx)

            # Pick a random feasible action
            final_action_idx = choice(allowable_actions_idxs)
            final_action, utils = veh_action_choices[veh_idx][final_action_idx]
            final_actions[veh_idx] = (final_action, utils)

            # Update inefasible action information
            for request in final_action.requests:
                consumed_requests.add(request)

        # Make sure all vehicles have a corresponding action
        for action in final_actions:  # type: ignore
            assert action is not None

        return final_actions

    def choose_actions_greedy(
        self,
        veh_action_choices: List[List[Tuple[Action, List[float]]]],
        is_training: bool = True,
        epoch_num: int = 1
    ) -> List[List[Tuple[Action, float]]]:
        """
        Steps:
        Add noise to each action util
        Repeat while until all vehicles not assigned:
            Find the largest scoring action.
            Assign that action to corresponding vehicle
            Remove corresponding requests/actions from all other vehicles.

            Make sure to not remove the null action.
        """

        # Define noise function for exploration
        def get_noise(action_id: int) -> float:
            # scale = 1 + (40 if 'x0,' in variable.get_name() else 10) / ((epoch_num + 1) * self.envt.NUM_VEHS)
            scale = 1 + (4000 if action_id==0 else 1000) / ((epoch_num + 1) * self.envt.NUM_VEHS)
            return uniform(0, scale) if is_training else 0

        #Add noise to scores and create new list
        veh_action_choices_util = []
        for veh_idx, scored_actions_per_veh in enumerate(veh_action_choices):
            veh_scored_actions_util = []
            for action_idx, (action, util) in enumerate(scored_actions_per_veh):
                noisy_util = util + get_noise(action_idx)
                veh_scored_actions_util.append((action, noisy_util, veh_idx))

            #Sort each vehicle's actions by decreasing score and add to list
            sorted_action_choices = sorted(veh_scored_actions_util,key=lambda x: x[1], reverse=True)
            veh_action_choices_util.append(sorted_action_choices)

        #Get a greedy assignment
        final_actions: List[Optional[Tuple[Action, float]]] = [None] * len(veh_action_choices)
        while len(veh_action_choices_util):
            #Select the vehicle with the largest first util (since actions are sorted by utils)
            largest_util = -1000000
            largest_ind = -1
            for rem_veh_id, scored_choices in enumerate(veh_action_choices_util):
                _, util, _ = scored_choices[0]
                if util>largest_util:
                    largest_util = util
                    largest_ind = rem_veh_id
            
            #Remove this vehicle from the options and save as final choice
            best_veh_actions = veh_action_choices_util.pop(largest_ind)
            best_action, best_util, veh_idx = best_veh_actions[0]
            final_actions[veh_idx] = (best_action, best_util)

            #Remove all actions with common requests
            for rem_idx, scored_choices in enumerate(veh_action_choices_util):
                veh_actions_rem = []
                for action_idx, (action,_,_) in enumerate(scored_choices):
                    for req in best_action.requests:
                        if req in action.requests:
                            veh_actions_rem.append(action_idx)
                            break
                for i in sorted(veh_actions_rem, reverse=True):
                    veh_action_choices_util[rem_idx].pop(i)
        
        # Make sure all vehicles have a corresponding action
        for action in final_actions:  # type: ignore
            assert action is not None

        return final_actions

    def choose_actions_random_greedy(
        self,
        veh_action_choices: List[List[Tuple[Action, List[float]]]],
        is_training: bool = True,
        epoch_num: int = 1
    ) -> List[List[Tuple[Action, float]]]:
        """
        Take greedy actions by taking a random order of vehicles
        Steps:
        Add noise to each action util
        Repeat while until all vehicles not assigned:
            Pick a random vehicle
            Assign its highest scoring action
            Find the largest scoring action.
            Remove corresponding requests/actions from all other vehicles.

            Make sure to not remove the null action.
        """

        # Define noise function for exploration
        def get_noise(action_id: int) -> float:
            # scale = 1 + (40 if 'x0,' in variable.get_name() else 10) / ((epoch_num + 1) * self.envt.NUM_VEHS)
            scale = 1 + (4000 if action_id==0 else 1000) / ((epoch_num + 1) * self.envt.NUM_VEHS)
            return uniform(0, scale) if is_training else 0

        #Add noise to scores and create new list
        veh_action_choices_util = []
        for veh_idx, scored_actions_per_veh in enumerate(veh_action_choices):
            veh_scored_actions_util = []
            for action_idx, (action, util) in enumerate(scored_actions_per_veh):
                noisy_util = util + get_noise(action_idx)
                veh_scored_actions_util.append((action, noisy_util, veh_idx))

            #Sort each vehicle's actions by decreasing score and add to list
            sorted_action_choices = sorted(veh_scored_actions_util,key=lambda x: x[1], reverse=True)
            veh_action_choices_util.append(sorted_action_choices)

        #Get a greedy assignment
        used_reqs = set()
        final_actions = [None] * len(veh_action_choices)
        # Get a random order of vehicles
        veh_ids = list(range(len(veh_action_choices)))
        shuffle(veh_ids)
        for veh_idx in veh_ids:
            #Assign the highest scoring action that does not use any of the used requests
            for action, util, _ in veh_action_choices_util[veh_idx]:
                if not any([req in used_reqs for req in action.requests]):
                    final_actions[veh_idx] = (action, util)
                    used_reqs.update(action.requests)
                    break
        # Make sure all vehicles have a corresponding action
        for action in final_actions:  # type: ignore
            assert action is not None

        return final_actions
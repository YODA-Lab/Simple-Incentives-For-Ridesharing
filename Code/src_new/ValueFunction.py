import functools

from Vehicle import Vehicle
from Action import Action
from Environment import Environment
from ReplayBuffer import PrioritizedReplayBuffer
from Experience import Experience
from CentralAgent import CentralAgent
from Request import Request
from Pricer import Pricer
from Customer import Customer
from utils import filter_actions_by_req, add_reward_to_score, get_flattened_scores, dict2mat, add_fairness_to_score

from typing import List, Tuple, Dict, Any, Callable

from abc import ABC, abstractmethod
from keras.utils.vis_utils import model_to_dot, plot_model
from keras.layers import Input, LSTM, Dense, Embedding, Masking, Concatenate, Lambda  # type: ignore
from keras.models import Model, load_model, clone_model  # type: ignore
from keras.backend import function as keras_function  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from keras.initializers import Constant  # type: ignore
from keras.optimizers import Adam  # type: ignore
from tensorflow.summary import FileWriter, histogram  # type: ignore
from tensorflow import Summary  # type: ignore
import tensorflow as tf
import numpy as np
from copy import deepcopy
from os.path import isfile, isdir
from os import makedirs
from functools import partial
from math import exp
import pickle

from random import shuffle, randint, random, uniform, choice
import time

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True

class ValueFunction(ABC):
    """docstring for ValueFunction"""

    def __init__(self, log_dir: str):
        super(ValueFunction, self).__init__()

        # Write logs
        log_dir = log_dir + type(self).__name__ + '/'
        if not isdir(log_dir):
            makedirs(log_dir)
        self.writer = FileWriter(log_dir)
        #For tag logging
        self.base_log_dir = log_dir
        self.log_dir = log_dir

    def add_to_logs(self, tag: str, value: float, step: int) -> None:
        summary = Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, step)
        # self.writer.flush() #Slows things down a lot. Use if immediate updates are needed
    
    def add_to_logs_mod(self, tags: str, values: float, step: int, mod: str) -> None:
        #Add a set of tags and values to logs and save in self.log_dir+mod subdirectory
        summary = Summary()
        for tag,value in zip(tags, values):
            summary.value.add(tag=tag, simple_value=value)
        log_dir = self.base_log_dir + mod + '/'
        if log_dir!=self.log_dir:
            self.writer.flush()
            if not isdir(log_dir):
                makedirs(log_dir)
            self.writer=FileWriter(log_dir)
            self.log_dir = log_dir
        self.writer.add_summary(summary, step)
    
    def log_histogram(self, tag, values, step, bins=1000):
        #src: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        # self.writer.flush()

    @abstractmethod
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        raise NotImplementedError

    @abstractmethod
    def get_future_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        raise NotImplementedError

    @abstractmethod
    def update(self, central_agent: CentralAgent):
        raise NotImplementedError

    @abstractmethod
    def remember(self, experience: Experience):
        raise NotImplementedError

class RewardPlusDelay(ValueFunction):
    """docstring for RewardPlusDelay"""

    def __init__(self, envt: Environment, DELAY_COEFFICIENT: float = 1e-3, log_dir='../logs/', is_discount=False):
        super(RewardPlusDelay, self).__init__(log_dir)
        self.envt = envt
        self.DELAY_COEFFICIENT = DELAY_COEFFICIENT
        self.is_discount = is_discount

    def _get_future_value(self, action: Action) -> float:
        assert action.new_path
        remaining_delay_bonus = self.DELAY_COEFFICIENT * action.new_path.total_delay

        return remaining_delay_bonus

    def _populate_values(self, experiences: List[Experience], score_fn: Callable[[Action], float]) -> List[List[Tuple[Action, float]]]:
        scored_actions_all_vehs: List[List[Tuple[Action, float]]] = []
        for experience in experiences:
            for feasible_actions in experience.feasible_actions_all_vehs:
                scored_actions: List[Tuple[Action, float]] = []
                for action in feasible_actions:
                    score = score_fn(action)
                    scored_actions.append((action, score))
                scored_actions_all_vehs.append(scored_actions)

        return scored_actions_all_vehs

    def get_future_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        score_fn = self._get_future_value
        scored_actions_all_vehs = self._populate_values(experiences, score_fn)

        return scored_actions_all_vehs

    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        future_val = self._get_future_value
        reward = self.envt.get_reward if self.is_discount else self.envt.get_revenue
        scored_actions_all_vehs = self._populate_values(experiences, lambda a: reward(a) + future_val(a))

        return scored_actions_all_vehs

    def update(self, *args, **kwargs):
        pass

    def remember(self, *args, **kwargs):
        pass


class ImmediateReward(RewardPlusDelay):
    """TODO: Document ImmediateReward"""

    def __init__(self, envt: Environment, log_dir='../logs/', is_discount=False):
        super(ImmediateReward, self).__init__(envt=envt, DELAY_COEFFICIENT=0, log_dir=log_dir, is_discount=is_discount)


class NeuralNetworkBased(ValueFunction):
    """docstring for NeuralNetwork"""

    def __init__(
        self,
        envt: Environment,
        pricer: Pricer,
        customer: Customer,
        load_model_loc: str,
        log_dir: str,
        BATCH_SIZE_FIT: int = 128,
        BATCH_SIZE_PREDICT: int = 8192,
        TARGET_UPDATE_TAU: float = 0.005
    ):
        super(NeuralNetworkBased, self).__init__(log_dir)

        # Initialise Constants
        self.envt = deepcopy(envt)
        self.pricer = pricer
        self.customer = customer
        self.BATCH_SIZE_FIT = BATCH_SIZE_FIT
        self.BATCH_SIZE_PREDICT = BATCH_SIZE_PREDICT
        self.TARGET_UPDATE_TAU = TARGET_UPDATE_TAU
        self.delay_factor = 0 #2
        self.DELAY_COEFFICIENT = 1e-3

        self._epoch_id = 0

        # Get Replay Buffer
        MIN_LEN_REPLAY_BUFFER = 1e6 / self.envt.NUM_VEHS
        epochs_in_episode = (self.envt.STOP_EPOCH - self.envt.START_EPOCH) / self.envt.EPOCH_LENGTH
        len_replay_buffer = max((MIN_LEN_REPLAY_BUFFER, epochs_in_episode))
        self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

        # Get NN Model
        self.model: Model = load_model(load_model_loc) if load_model_loc else self._init_NN(self.envt.NUM_LOCATIONS)

        # Define Loss and Compile
        opt = Adam(amsgrad=True, clipnorm=1.0)
        self.model.compile(optimizer=opt, loss='mean_squared_error')
        # self.model.compile(optimizer=opt, loss=tf.compat.v1.losses.Huber(delay = 1.0))
        
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

        # Get target-NN
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Define soft-update function for target_model_update
        self.update_target_model = self._soft_update_function(self.target_model, self.model)

    def _soft_update_function(self, target_model: Model, source_model: Model) -> keras_function:
        target_weights = target_model.trainable_weights
        source_weights = source_model.trainable_weights

        updates = []
        for target_weight, source_weight in zip(target_weights, source_weights):
            updates.append((target_weight, self.TARGET_UPDATE_TAU * source_weight + (1. - self.TARGET_UPDATE_TAU) * target_weight))

        return keras_function([], [], updates=updates)

    @abstractmethod
    def _init_NN(self, num_locs: int):
        raise NotImplementedError()

    @abstractmethod
    def _format_input_batch(self, vehs: List[List[Vehicle]], current_time: float, num_requests: int):
        raise NotImplementedError

    def _get_vehs_post_actions(self, experience: Experience) -> List[List[Vehicle]]:
        all_vehs_post_actions: List[List[Vehicle]] = []
        for veh, feasible_actions in zip(experience.vehs, experience.feasible_actions_all_vehs): #New
            vehs_post_actions: List[Vehicle] = []
            for action in feasible_actions:
                veh_post_action = Vehicle(veh.id, deepcopy(veh.position))
                assert action.new_path
                veh_post_action.path = action.new_path #Skipping the deepcopy here saves a lot of time
                veh_post_action.earning = veh.earning
                veh_post_action.discounted_value = veh.discounted_value
                vehs_post_actions.append(veh_post_action)

            all_vehs_post_actions.append(vehs_post_actions)


        return all_vehs_post_actions

    def _flatten_NN_input(self, NN_input: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], List[int]]:
        shape_info: List[int] = []

        for key, value in NN_input.items():
            # Remember the shape information of the inputs
            if not shape_info:
                cumulative_sum = 0
                shape_info.append(cumulative_sum)
                for idx, list_el in enumerate(value):
                    cumulative_sum += len(list_el)
                    shape_info.append(cumulative_sum)

            # Reshape
            NN_input[key] = np.array([element for array in value for element in array])

        return NN_input, shape_info

    def _reconstruct_NN_output(self, NN_output: np.ndarray, shape_info: List[int]) -> List[List[int]]:
        # Flatten output
        NN_output = NN_output.flatten()

        # Reshape
        assert shape_info
        output_as_list = []
        for idx in range(len(shape_info) - 1):
            start_idx = shape_info[idx]
            end_idx = shape_info[idx + 1]
            list_el = NN_output[start_idx:end_idx].tolist()
            output_as_list.append(list_el)

        return output_as_list

    def _format_experiences(self, experiences: List[Experience], is_prev: bool = False) -> Tuple[Dict[str, np.ndarray], List[int]]:
        action_inputs_all_vehs = None
        for experience in experiences:
            # If experience has been formatted previously, use it
            if (is_prev in experience.representation):
                batch_input = deepcopy(experience.representation[is_prev])
            # Else, format it
            else:
                if is_prev:
                    vehs_post_actions = [[veh] for veh in experience.vehs_prev]
                else:
                    vehs_post_actions = self._get_vehs_post_actions(experience)
                # Get formatted input
                batch_input = self._format_input_batch(vehs_post_actions, experience.time, experience.num_requests)
                # Save a copy for future use
                experience.representation[is_prev] = deepcopy(batch_input)

            # Add formatted experience to data struct
            if action_inputs_all_vehs is None:
                action_inputs_all_vehs = batch_input
            else:
                for key, value in batch_input.items():
                    action_inputs_all_vehs[key].extend(value)
        assert action_inputs_all_vehs is not None

        return self._flatten_NN_input(action_inputs_all_vehs)

    def _get_values(self, experiences: List[Experience], get_score: Callable[[Action, float], float], network: Model = None, is_prev: bool = False) -> List[List[Tuple[Action, float]]]:
        # Format experiences
        action_inputs_all_vehs, shape_info = self._format_experiences(experiences, is_prev)

        # Score experiences
        if (network is None):
            expected_future_values_all_vehs = self.model.predict(action_inputs_all_vehs, batch_size=self.BATCH_SIZE_PREDICT)
        else:
            expected_future_values_all_vehs = network.predict(action_inputs_all_vehs, batch_size=self.BATCH_SIZE_PREDICT)

        # Format NN output
        expected_future_values_all_vehs = self._reconstruct_NN_output(expected_future_values_all_vehs, shape_info)

        # Reformat experiences
        feasible_actions_all_vehs = [feasible_actions for experience in experiences for feasible_actions in experience.feasible_actions_all_vehs]

        scored_actions_all_vehs: List[List[Tuple[Action, float]]] = []
        for expected_future_values, feasible_actions in zip(expected_future_values_all_vehs, feasible_actions_all_vehs):
            scored_actions = [(action, get_score(action, value)+ self.delay_factor*self.DELAY_COEFFICIENT * action.new_path.total_delay) for action, value in zip(feasible_actions, expected_future_values)]
            # scored_actions = [(action, get_score(action, value)) for action, value in zip(feasible_actions, expected_future_values)]
            scored_actions_all_vehs.append(scored_actions)
        return scored_actions_all_vehs

    def get_future_value(self, experiences: List[Experience], network: Model = None, is_prev: bool = False) -> List[List[Tuple[Action, float]]]:
        # Score is just the future value
        def get_score(self, action: Action, value: float) -> float:
            return self.envt.GAMMA * value
        score_fn = partial(get_score, self)

        scored_actions_all_vehs = self._get_values(experiences, score_fn, network, is_prev)
        return scored_actions_all_vehs

    def get_value(self, experiences: List[Experience], network: Model = None, is_prev: bool = False) -> List[List[Tuple[Action, float]]]:
        # Score is the sum of future value and immediate reward
        def get_score(self, action: Action, value: float) -> float:
            reward = self.envt.get_reward(action)
            return reward + self.envt.GAMMA * value
        score_fn = partial(get_score, self)

        scored_actions_all_vehs = self._get_values(experiences, score_fn, network, is_prev)
        return scored_actions_all_vehs

    def remember(self, experience: Experience):
        self.replay_buffer.add(experience)

    def update(self, central_agent: CentralAgent, num_samples: int = 64):
        # Check if replay buffer has enough samples for an update
        num_min_train_samples = max(1e5 / self.envt.NUM_VEHS, num_samples)
        # num_min_train_samples = max(5e3 / self.envt.NUM_VEHS, num_samples)
        if (num_min_train_samples > len(self.replay_buffer)):
            return
        print("Updating")

        # SAMPLE FROM REPLAY BUFFER
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # TODO: Implement Beta Scheduler
            beta = 0.4 + 0.5 * (1 - exp(- self._epoch_id / 200.0))
            experiences, weights, batch_idxes = self.replay_buffer.sample(num_samples, beta)
            weights = np.array([[weight] * self.envt.NUM_VEHS for weight in weights]).flatten()
        else:
            experiences = self.replay_buffer.sample(num_samples)
            weights = None

        # UPDATE POLICY BASED ON SAMPLES OF PAST EXPERIENCE
        #   FOR EACH SAMPLE, DETERMINE THE BEST ACTION
        final_action_experiences: List[Experience] = []
        penalty = 0
        for experience in experiences:
            # Score experiences based on future value
            scored_actions_all_vehs = self.get_future_value([experience])  # type: ignore

            # Price feasible actions
            current_requests = list(set([request
                                         for actions_per_veh in scored_actions_all_vehs
                                         for action, _ in actions_per_veh
                                         for request in action.requests]))
            req_to_price = self.pricer.get_prices(current_requests, scored_actions_all_vehs, self.customer)

            # Show prices to customers and see how which prices were accepted
            # accepted_reqs = [request for request, price in req_to_price.items() if self.customer.is_accept(request, price)]
            accepted_reqs = [request for request, price in req_to_price.items()] 
            for request in accepted_reqs:
                request.price = req_to_price[request]  # setting price attributes for accepted requests
            
            # Add the immediate reward (price) to the future value to get the final score
            scored_actions_all_vehs = add_reward_to_score(scored_actions_all_vehs, self.envt)

            # Choose actions for each vehicle
            scored_final_actions = central_agent.choose_actions(scored_actions_all_vehs, is_training=False)

            final_actions = [[action] for action, _ in scored_final_actions]
            final_action_experiences.append(Experience(deepcopy(experience.vehs_prev), deepcopy(experience.vehs), final_actions, experience.time, experience.num_requests, deepcopy(experience.SR_flat), experience.mean_SR))

        #   ESTIMATE TD-TARGET FOR BEST ACTION USING TARGET NETWORK
        scored_actions_all_vehs = self.get_value(final_action_experiences, network=self.target_model)

        #   UPDATE NN BASED ON TD-TARGETS
        td_targets = np.array(get_flattened_scores(scored_actions_all_vehs)).reshape((-1, 1))
        action_inputs_all_vehs_prev, _ = self._format_experiences(experiences, is_prev=True)

        history = self.model.fit(action_inputs_all_vehs_prev, td_targets, epochs=100, validation_split=0.1, batch_size=self.BATCH_SIZE_FIT, sample_weight=weights, callbacks=self.callbacks)

        # DO BOOKKEEPING
        # Write to logs
        loss = history.history['loss'][-1]
        self.add_to_logs('loss', loss, self._epoch_id)
        val_loss = history.history['val_loss'][-1]
        self.add_to_logs('val loss', val_loss, self._epoch_id)

        # Update weights of replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # Calculate the TD-Error after the update
            scored_actions_all_vehs_prev = self.get_future_value(experiences, is_prev=True)
            future_val_prev = get_flattened_scores(scored_actions_all_vehs_prev)

            scored_actions_all_vehs = self.get_value(final_action_experiences)
            future_val = get_flattened_scores(scored_actions_all_vehs)

            # NOTE: Assumes that there are self.envt.NUM_VEHS entries in each experience
            squared_error = (np.array(future_val_prev) - np.array(future_val)) ** 2
            mean_squared_error = squared_error.reshape((num_samples, self.envt.NUM_VEHS)).mean(axis=1)
            # print(squared_error)
            # Update priorities based on TD-Error
            
            new_batch_idxes = []
            new_errors = []
            for batch_id, error in zip(batch_idxes, list(mean_squared_error)):
                if error!=0:
                    new_batch_idxes.append(batch_id)
                    new_errors.append(error)
            self.replay_buffer.update_priorities(new_batch_idxes, new_errors)
            # self.replay_buffer.update_priorities(batch_idxes, list(mean_squared_error))

        # Soft update target_model based on the learned model
        self.update_target_model([])

        self._epoch_id += 1


class PathBasedNN(NeuralNetworkBased):

    def __init__(
        self,
        envt: Environment,
        pricer: Pricer,
        customer: Customer,
        load_model_loc: str = '',
        log_dir: str = '../logs/'
    ):
        super(PathBasedNN, self).__init__(envt, pricer, customer, load_model_loc, log_dir)

    def _init_NN(self, num_locs: int) -> Model:
        # DEFINE NETWORK STRUCTURE
        # Check if there are pretrained embeddings
        if (isfile(self.envt.DATA_DIR + 'embedding_weights.pkl')):
            weights = pickle.load(open(self.envt.DATA_DIR + 'embedding_weights.pkl', 'rb'))
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding', embeddings_initializer=Constant(weights[0]), trainable=False)
        else:
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding')

        # Get path and current locations' embeddings
        path_location_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32', name='path_location_input')
        print(self.envt.MAX_CAPACITY * 2 + 1, self.envt.NUM_LOCATIONS + 1)
        path_location_embed = location_embed(path_location_input)

        # Get associated delay for different path locations
        delay_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1), name='delay_input')
        delay_masked = Masking(mask_value=-1)(delay_input)

        # Get entire path's embedding
        path_input = Concatenate()([path_location_embed, delay_masked])
        path_embed = LSTM(200, go_backwards=False)(path_input)

        # Get current time's embedding
        current_time_input = Input(shape=(1,), name='current_time_input')
        current_time_embed = Dense(100, activation='elu', name='time_embedding')(current_time_input)

        # Get embedding for other agents
        other_vehs_input = Input(shape=(1,), name='other_vehs_input')

        # Get embedding for number of requests
        num_requests_input = Input(shape=(1,), name='num_requests_input')
        print("Num GPUs Available: ", tf.test.gpu_device_name())
        # Get Embedding for the entire thing
        state_embed = Concatenate()([path_embed, current_time_embed, other_vehs_input, num_requests_input])
        state_embed = Dense(300, activation='elu', name='state_embed_1')(state_embed)
        state_embed = Dense(300, activation='elu', name='state_embed_2')(state_embed)

        # Get predicted Value Function
        output = Dense(1, name='output')(state_embed)

        model = Model(inputs=[path_location_input, delay_input, current_time_input, other_vehs_input, num_requests_input], outputs=output)

        return model

    def _format_input(self, veh: Vehicle, current_time: float, num_requests: float, num_other_vehs: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        # Normalising Inputs
        current_time_input = (current_time - self.envt.START_EPOCH) / (self.envt.STOP_EPOCH - self.envt.START_EPOCH)
        num_requests_input = num_requests / self.envt.NUM_VEHS
        num_other_vehs_input = num_other_vehs / self.envt.NUM_VEHS

        # Getting path based inputs
        location_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32')
        delay_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1)) - 1

        # Adding current location
        location_order[0] = veh.position.next_location + 1
        delay_order[0] = 1

        for idx, node in enumerate(veh.path.request_order):
            if (idx >= 2 * self.envt.MAX_CAPACITY):
                break

            location, deadline = veh.path.get_info(node)
            visit_time = node.expected_visit_time

            location_order[idx + 1] = location + 1
            delay_order[idx + 1, 0] = (deadline - visit_time) / Request.MAX_DROPOFF_DELAY  # normalising

        return location_order, delay_order, current_time_input, num_requests_input, num_other_vehs_input

    def _format_input_batch(self, all_vehs_post_actions: List[List[Vehicle]], current_time: float, num_requests: int) -> Dict[str, Any]:
        input: Dict[str, List[Any]] = {"path_location_input": [], "delay_input": [], "current_time_input": [], "other_vehs_input": [], "num_requests_input": []}

        # find number of other vehicles j that can get to vehicle i within MAX_PICKUP_DELAY 
        # Efficient numpy computation, works 10x faster than the for loop below
        veh_positions = [veh_post_actions[0].position.next_location for veh_post_actions in all_vehs_post_actions]
        num_nearby_vehicles = np.sum(self.envt.travel_times[veh_positions][:, veh_positions]< Request.MAX_PICKUP_DELAY, axis=0)        
        # print("Time taken to calculate num_nearby_vehicles: ", time.time() - total_start)                    

        # Format all the other inputs
        for i,veh_post_actions in enumerate(all_vehs_post_actions):
            current_time_input = []
            num_requests_input = []
            path_location_input = []
            delay_input = []
            other_vehs_input = []

            # Get number of surrounding vehicles
            current_veh = veh_post_actions[0]
            num_other_vehs = num_nearby_vehicles[i]

            for veh in veh_post_actions:
                # Get formatted output for the state
                location_order, delay_order, current_time_scaled, num_requests_scaled, num_other_vehs_scaled = self._format_input(veh, current_time, num_requests, num_other_vehs)

                current_time_input.append(num_requests_scaled)
                num_requests_input.append(num_requests)
                path_location_input.append(location_order)
                delay_input.append(delay_order)
                other_vehs_input.append(num_other_vehs_scaled)

            input["current_time_input"].append(current_time_input)
            input["num_requests_input"].append(num_requests_input)
            input["delay_input"].append(delay_input)
            input["path_location_input"].append(path_location_input)
            input["other_vehs_input"].append(other_vehs_input)

        return input

class FairNNRider(PathBasedNN):
    #Dummy class for logging naveen ijcai paper fairness results
    def __init__(
        self,
        envt: Environment,
        pricer: Pricer,
        customer: Customer,
        load_model_loc: str = '',
        log_dir: str = '../logs/'
    ):
        super(FairNNRider, self).__init__(envt, pricer, customer, load_model_loc, log_dir)
    
    def _format_input_batch(self, all_vehs_post_actions: List[List[Vehicle]], current_time: float, num_requests: int) -> Dict[str, Any]:
        input: Dict[str, List[Any]] = {"path_location_input": [], "delay_input": [], "current_time_input": [], "other_agents_input": [], "num_requests_input": []}
        #Need to define this again just because they use a different name for other vehs input

        # Format all the other inputs
        for veh_post_actions in all_vehs_post_actions:
            current_time_input = []
            num_requests_input = []
            path_location_input = []
            delay_input = []
            other_vehs_input = []

            # Get number of surrounding vehicles
            current_veh = veh_post_actions[0]
            num_other_vehs = 0
            for other_vehs_post_actions in all_vehs_post_actions:
                other_veh = other_vehs_post_actions[0]
                if (self.envt.get_travel_time(other_veh.position.next_location, current_veh.position.next_location) < Request.MAX_PICKUP_DELAY):
                    num_other_vehs += 1

            for veh in veh_post_actions:
                # Get formatted output for the state
                location_order, delay_order, current_time_scaled, num_requests_scaled, num_other_vehs_scaled = self._format_input(veh, current_time, num_requests, num_other_vehs)

                current_time_input.append(num_requests_scaled)
                num_requests_input.append(num_requests)
                path_location_input.append(location_order)
                delay_input.append(delay_order)
                other_vehs_input.append(num_other_vehs_scaled)

            input["current_time_input"].append(current_time_input)
            input["num_requests_input"].append(num_requests_input)
            input["delay_input"].append(delay_input)
            input["path_location_input"].append(path_location_input)
            input["other_agents_input"].append(other_vehs_input)

        return input

class FairNNDriver(FairNNRider):
    #Dummy class for logging if NN value isnt added to score. Only vary beta, alpha = 1
    def __init__(
        self,
        envt: Environment,
        log_dir: str = '../logs/',
        is_discount = False
    ):
        super(FairNNDriver, self).__init__(envt, log_dir = log_dir, is_discount=is_discount)

class Random(ImmediateReward):
    #Dummy class for logging random assignment results
    def __init__(
        self,
        envt: Environment,
        log_dir: str = '../logs/',
        is_discount = False
    ):
        super(Random, self).__init__(envt, log_dir, is_discount=is_discount)

class RandomGreedyNN(PathBasedNN):
    #Dummy class for logging if random greedy is used 
    def __init__(
        self,
        envt: Environment,
        pricer: Pricer,
        customer: Customer,
        load_model_loc: str = '',
        log_dir: str = '../logs/'
    ):
        super(RandomGreedyNN, self).__init__(envt, pricer, customer, load_model_loc, log_dir)
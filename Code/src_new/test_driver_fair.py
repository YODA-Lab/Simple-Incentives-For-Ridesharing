from Environment import Environment, MapEnvironment
from CentralAgent import CentralAgent
from Vehicle import Vehicle
from Oracle import Oracle
from ValueFunction import *
from Experience import Experience
from Request import Request
from Customer import *
from Pricer import *
from utils import filter_actions_by_req, add_reward_to_score, add_fairness_to_score, add_driver_fairness_to_score, gini, get_driver_earning
from BasePrice import *

from typing import List, Optional

# import pdb
from copy import deepcopy
import argparse
from random import seed
import pickle
import os


# set seed
import random
import numpy as np
from tensorflow import set_random_seed
import time

seed_num = int(round(time.time()))
seed_num =139325
random.seed(seed_num)
np.random.seed(seed_num)
set_random_seed(seed_num)

#For libiomp5 multiple initialisation error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run_epoch(envt: Environment,
              oracle: Oracle,
              central_agent: CentralAgent,
              value_function: ValueFunction,
              customer: Customer,
              pricer: Pricer,
              DAY: int,
              is_training: bool,
              vehs_predefined: Optional[List[Vehicle]] = None,
              TRAINING_FREQUENCY: int = 10,
              is_testing: bool = False,
              solver='ILP'):

    # INITIALISATIONS
    total_start = time.time()
    # Initialising vehs
    if vehs_predefined is not None:
        vehs: List[Vehicle] = deepcopy(vehs_predefined)
    else:
        initial_states: List[int] = envt.get_initial_states(envt.NUM_VEHS, is_training)
        vehs = [Vehicle(veh_idx, initial_state) for veh_idx, initial_state in enumerate(initial_states)]

    # ITERATING OVER TIMESTEPS
    print("DAY: {}".format(DAY))
    request_generator = envt.get_request_batch(day=DAY)
    total_value_generated = 0.0
    num_total_requests = 0
    vehs_prev = deepcopy(vehs)
    verbose=False #If extra output needed
    while True:
        # Get new requests
        try:
            current_requests = next(request_generator)
            prob = 1
            if prob!=1:
                print(len(current_requests), end=' ')
                #Subsample requests
                select = np.random.rand(len(current_requests))<=prob
                current_requests = [req for req, sel in zip(current_requests, select) if sel]
                print(len(current_requests))
            if verbose: print()
            print(f"Current time: {envt.current_time // 3600:02d}:{(envt.current_time // 60) % 60:02d}, DAY: {DAY}, {args.valuefunction}({args.alpha}, {args.beta}) SR pair: Mean={envt.mean_pair_sr}, Min={envt.min_pair_sr}",end='\r')
            if verbose: print()
            if verbose: print("Number of new requests: {}".format(len(current_requests)))
            num_total_requests += len(current_requests)
            current_requests.extend(envt.carryover_reqs)
            if verbose: print("Number of total requests: {}".format(len(current_requests)))
        except StopIteration:
            break

        #price requests
        for request in current_requests:
            request.price = request.base_price
        if verbose: print('Priced requests')

        # Choose actions for each request/vehicle

        #General Form:
        feasible_actions_all_vehs = oracle.get_feasible_actions(vehs, current_requests)
        
        SR = envt.source_sr if args.fairtype=='src' else envt.pair_sr
        SR_mat = dict2mat(SR, key_order=envt.keys_list, select_index=0)
        envt.SR_flat = [sr for row in SR_mat for sr in row]
        envt.mean_SR = envt.mean_source_sr if args.fairtype=='src' else envt.mean_pair_sr
        experience = Experience(vehs_prev, deepcopy(vehs), feasible_actions_all_vehs, envt.current_time, len(current_requests), envt.SR_flat, envt.mean_SR)
        scored_actions_all_vehs = value_function.get_future_value([experience])
        if verbose: print('Scored actions')
        
        scored_actions_all_vehs = add_reward_to_score(scored_actions_all_vehs, envt)


        if verbose: print('Added bonus and reward actions')
        #ILP 
        if solver=='ILP':
            scored_final_actions = central_agent.choose_actions(scored_actions_all_vehs, is_training=is_training, epoch_num=envt.num_days_trained)
        
        # Greedy #Current implementation is even less efficient than ILP
        if solver=='greedy':
            scored_final_actions = central_agent.choose_actions_greedy(scored_actions_all_vehs, is_training=is_training, epoch_num=envt.num_days_trained)

        #Random greedy #is an order of magnitude faster than ILP
        if solver=='random_greedy':
            scored_final_actions = central_agent.choose_actions_random_greedy(scored_actions_all_vehs, is_training=is_training, epoch_num=envt.num_days_trained)

        # Random 
        if solver=='random':
            scored_final_actions = central_agent._choose_actions_random(scored_actions_all_vehs)
        #++++++++++++++++++++++++++++++++++++++++++++++
        
        accepted_reqs = []
        for action, _ in scored_final_actions:
            accepted_reqs.extend(list(action.requests))

        if verbose: print("got final actions")

        dropped_reqs, carryover_reqs = envt.get_dropped_requests(current_requests, scored_final_actions, carryover = False)#not(args.no_carryover)) #ONLY WITH NO CARRYOVER
        envt.carryover_reqs = deepcopy(carryover_reqs)
        if verbose: print(f'{len(dropped_reqs)} requests dropped')
        if verbose: print(f'{len(carryover_reqs)} requests carried over')

        # Assign final actions to vehicles
        for veh_idx, (action, _) in enumerate(scored_final_actions):
            assert action.new_path  # sanity check
            vehs[veh_idx].path = deepcopy(action.new_path)
            #Update driver earning. Assume it is proportional to the request price
            earning = get_driver_earning(action)
            vehs[veh_idx].earning+=earning
            vehs[veh_idx].discounted_value*=vehs[veh_idx].discount
            vehs[veh_idx].discounted_value+=earning

        # Calculate reward for selected actions
        rewards = []
        for action, _ in scored_final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += envt.get_revenue(action)

        if verbose: print(f"Number of requests served: {len([request for action, _ in scored_final_actions for request in action.requests])}")
        if verbose: print(f"Revenue for epoch: {sum(envt.get_revenue(action) for action, _ in scored_final_actions)}")
        if verbose: print(f"Reward for epoch: {sum(rewards)}")
        if verbose: print()

        # Carryover requests and update gini
        envt.update_all(scored_final_actions, vehs, day=DAY, dropped_requests = dropped_reqs)
        
        # Update
        if (is_training):
            value_function.remember(experience)
            
            # Update value function every TRAINING_FREQUENCY timesteps
            if ((int(envt.current_time) / int(envt.EPOCH_LENGTH)) % TRAINING_FREQUENCY == TRAINING_FREQUENCY - 1):
                value_function.update(central_agent)
                print(f"Current time: {envt.current_time // 3600:02d}:{(envt.current_time // 60) % 60:02d}, DAY: {DAY}, {args.valuefunction}({args.alpha}, {args.beta}) SR pair: Mean={envt.mean_pair_sr}, Min={envt.min_pair_sr}")

        # Sanity check
        for veh in vehs:
            assert envt.has_valid_path(veh)
        if verbose: print("Asserted sanity")
        # Writing statistics to logs
        avg_capacity = sum([veh.path.current_capacity for veh in vehs]) / envt.NUM_VEHS
        stats = {
                'rewards':sum(rewards),
                'Cumulative_rewards': total_value_generated,
                'avg_capacity': avg_capacity,
                'SR': envt.service_rate,
                'Pair_SR_mean': envt.mean_pair_sr,
                'Pair_SR_min': envt.min_pair_sr,
                'Pair_SR_StD': envt.pair_sr_std,
                'Pair_SR_Gini': envt.gini_pair_sr,
                'Source_SR_mean': envt.mean_source_sr,
                'Source_SR_min': envt.min_source_sr,
                'Source_SR_Gini': envt.gini_source_sr,
                "Driver_Gini": envt.driver_gini,
                "Driver_Min": envt.driver_min,
                "Driver_mean": envt.driver_avg
        }
        value_function.add_to_logs_mod(stats.keys(), stats.values(), envt.current_time//60, ("Test_" if is_testing else "") + str(DAY) + 'Day'+str(envt.num_days_trained))

        if (envt.current_time // 60) % 10 ==0:
            value_function.writer.flush()

        if verbose: print("Added to logs")
        # Simulate the passing of time
        vehs_prev = deepcopy(vehs)
        envt.simulate_motion(vehs, current_requests)
    
        if verbose: print("Simulated passing of time")
        if verbose: print("Cumulative revenue: {}".format(total_value_generated))
    total_end = time.time()

    # Printing statistics for current epoch
    print('\nTotal Revenue: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))

    if is_testing:
        tag_suffix = f'_{args.tag}' if args.tag else ''
        value_function_name = type(value_function).__name__ + '/'
        expr_name = f'SI/{args.numvehs}veh_{args.capacity}cap/DAY{DAY}({args.starthour}-{args.endhour}{tag_suffix})/alpha_d{args.alpha_d}/delta{args.delta}/alpha{args.alpha}/beta{args.beta}/{args.fairtype}{value_function_name}'
        logloc = f"../logs/{args.numvehs}veh_{args.capacity}cap_{args.environment}{args.starthour}-{args.endhour}{args.tag if args.tag else ''}/alpha_d{args.alpha_d}/delta{args.delta}/alpha{args.alpha}/beta{args.beta}/{args.fairtype}{value_function_name}"
        folder_path = f'../../Runs/{expr_name}'
        save_run = True
        if save_run:
            os.makedirs(folder_path, exist_ok=True)
            pickle.dump([envt.service_rate, envt.mean_pair_sr, envt.pair_sr], open(os.path.join(folder_path,"pair_sr.pkl"), "wb"))
            earnings = np.array([v.earning for v in vehs])
            summarystats = {"SR":envt.service_rate, 
                            "Mean pair SR":envt.mean_pair_sr,
                            "Min pair SR": envt.min_pair_sr,
                            "Pair SR":envt.pair_sr,
                            "Pair SR Overall":envt.pair_sr_overall, 
                            "Pair SR gini": envt.gini_pair_sr,
                            "Mean source SR": envt.mean_source_sr,
                            "Min source SR":envt.min_source_sr,
                            "Source SR Overall": envt.source_sr_overall,
                            "Source SR": envt.source_sr,
                            "Source SR gini": envt.gini_source_sr,
                            "Rewards":total_value_generated,
                            "Trips Per Driver": earnings,
                            "TPD gini": gini(earnings),
                            "TPD min": min(earnings),
                            'runtime': total_end-total_start,
                            } 
            for k,v in vars(args).items():
                summarystats[k] = v
            pickle.dump(summarystats, open(os.path.join(folder_path,"summary.pkl"), "wb"))
            pickle.dump(summarystats, open(os.path.join(logloc,"summary.pkl"), "wb"))

    return total_value_generated


if __name__ == '__main__':
    # pdb.set_trace()

    # PARSE COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--capacity', type=int, default=4)
    parser.add_argument('-n', '--numvehs', type=int, default=1000)
    parser.add_argument('-tr', '--trainepochs', type=int, default=0)
    parser.add_argument('-s', '--starthour', type=int, default=18)
    parser.add_argument('-e', '--endhour', type=int, default=19)
    parser.add_argument('-v', '--valuefunction', type=str, default='rewardplusdelay')
    parser.add_argument('-m', '--modellocation', type=str, default='')
    parser.add_argument('-l', '--lamda', type=float, default=0., help='Weight of FairNN fairness term')
    parser.add_argument('-d', '--delta', type=float, default=0., help='Weight of driver fairness term')
    parser.add_argument('-b', '--beta', type=float, default=0., help='Weight of passenger fairness bonus term')
    parser.add_argument('-a', '--alpha', type=float, default=0., help='1-> SIP, 0->SIP(+)')
    parser.add_argument('-ad', '--alpha_d', type=float, default=0., help='1-> SID, 0->SID(+)')
    parser.add_argument('-ev', '--environment', type=str, default='ny')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('-f', '--filtered', action='store_true', default=False, help='whether to filter out neighborhoods')
    parser.add_argument('-sol', '--solver', type=str, default='ILP')
    parser.add_argument('-ftype', '--fairtype', type=str, default='', help='Choose between zone pair fairness (pair) and source zone fairness (src)')
    args = parser.parse_args()

    # CONSTANTS
    START_HOUR: int = args.starthour
    END_HOUR: int = args.endhour
    NUM_EPOCHS: int = args.trainepochs
    TRAINING_DAYS: List[int] = list(range(3, 11))
    VALID_DAYS: List[int] = [2]
    VALID_FREQ: int = 2
    SAVE_FREQ: int = VALID_FREQ
    TEST_DAYS: List[int] = [11]  # list(range(11, 16))
    pickup_delay = 300
    decisioninterval = 60
    Request.MAX_PICKUP_DELAY = pickup_delay
    Request.MAX_DROPOFF_DELAY = 2 * pickup_delay
    GAMMA = (1 - (0.1 * 60 / decisioninterval))
    LOG_DIR: str = f"../logs/{args.numvehs}veh_{args.capacity}cap_{args.environment}{args.starthour}-{args.endhour}{args.tag if args.tag else ''}/alpha_d{args.alpha_d}/delta{args.delta}/alpha{args.alpha}/beta{args.beta}/{args.fairtype}"

    # INITIALISATIONS
    customer: Customer = AlwaysAcceptCustomer()  # type: ignore
    pricer: Pricer = BasePricePricer()  # type: ignore
    ENVIRONMENT = {
        'ny': partial(MapEnvironment, filtered=args.filtered, DATA_DIR = '../data/ny/'),
        'conworld': partial(MapEnvironment, filtered=args.filtered, DATA_DIR = '../data/conworld/'),
    }
    NYENVT = ENVIRONMENT[args.environment]
    base_price: BasePrice = FlatPrice()  # type: ignore

    fairness_target=''
    if (args.alpha!=0 and args.beta!=0):
        fairness_target = 'areq'
    elif (args.alpha==0 and args.beta!=0):
        fairness_target = '+req'
    start_day = TEST_DAYS[0]
    
    envt = NYENVT(args.numvehs, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600, MAX_CAPACITY=args.capacity, EPOCH_LENGTH=60, BASE_PRICE=base_price, GAMMA=GAMMA, VALUE_FUNCTION=args.valuefunction, START_DAY=start_day, BETA=args.beta, ALPHA=args.alpha, DELTA=args.delta, FAIRNESS_TARGET=fairness_target, FAIRTYPE=args.fairtype)
    envt.lamda = args.lamda
    envt.plus_driver = False
    envt.alpha_d = args.alpha_d

    if (args.alpha_d==0 and args.delta!=0):
        print("Plus drivers only")
        envt.plus_driver = True

    print('plus_driver: ',envt.plus_driver, ' fairtype: ', envt.fairness_target )
    oracle = Oracle(envt)
    central_agent = CentralAgent(envt)
    solver = args.solver
    if args.valuefunction=='random_greedy':
        solver = 'random_greedy'

    #ADD TO THE ONES BELOW AS WELL AS TO ValueFunction.py TO LOG INTO THE CORRECT DIRECTORY WITH THE CORRECT NAME
    VALUE_FUNCTIONS = {
        'pathnn': partial(PathBasedNN, envt, pricer, customer, load_model_loc=args.modellocation),
        'fairnn_rider': partial(FairNNRider, envt, pricer, customer, load_model_loc=args.modellocation),
        'fairnn_driver': partial(FairNNDriver, envt, pricer, customer, load_model_loc=args.modellocation),
        'rewardplusdelay': partial(RewardPlusDelay, envt, is_discount=False),
        'immediatereward': partial(ImmediateReward, envt, is_discount=False),
        'random': partial(Random, envt, is_discount=False),
        'random_greedy':partial(RandomGreedyNN, envt, pricer, customer, load_model_loc=args.modellocation),
    }
    print(LOG_DIR)
    value_function: ValueFunction = VALUE_FUNCTIONS[args.valuefunction](log_dir=LOG_DIR)  # type: ignore          
    # TEST
    # Initialising vehicles
    initial_states = envt.get_initial_states(envt.NUM_VEHS, is_training=False)
    vehs_predefined = [Vehicle(veh_idx, initial_state) for veh_idx, initial_state in enumerate(initial_states)]

    envt.reset(flush=True)
    envt_test = envt
    oracle_test = oracle
    central_agent_test = central_agent
    for day in TEST_DAYS:
        start_time = time.time()
        total_requests_served = run_epoch(envt_test, oracle_test, central_agent_test, value_function, customer, pricer, day, is_training=False, vehs_predefined=vehs_predefined, is_testing=True, solver=solver)
        end_time = time.time()
        print(f"Day {day} took {end_time - start_time} seconds")
        print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
        value_function.add_to_logs('test_requests_served', total_requests_served, envt_test.num_days_trained)
        value_function.add_to_logs('test_time', end_time - start_time, envt_test.num_days_trained)
        envt.num_days_trained += 1
    print(args)
    envt.reset(flush=True)


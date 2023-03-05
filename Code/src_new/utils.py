from typing import List, Tuple, Callable, Optional

from functools import partial
import numpy as np
from copy import deepcopy

class IncorrectUsageError(Exception):
    """Custom exception that's called when a function is used incorrectly."""
    pass


def _filter_actions(
    scored_actions_all_vehs,
    process_fn
):
    new_scored_actions_all_vehs= []
    for scored_actions_per_veh in scored_actions_all_vehs:
        new_scored_actions = []
        for action, score in scored_actions_per_veh:
            new_scored_action = process_fn(action, score)
            if new_scored_action is not None:
                new_scored_actions.append(new_scored_action)
        new_scored_actions_all_vehs.append(new_scored_actions)

    return new_scored_actions_all_vehs


def filter_actions_by_req(
    scored_actions_all_vehs,
    rejected_reqs
):
    def process_fn(action, score: float):
        has_rejected_req = False
        for request in action.requests:
            if request in rejected_reqs:
                has_rejected_req = True
                break
        if has_rejected_req:
            return None
        else:
            return (action, score)

    new_scored_actions_all_vehs = _filter_actions(scored_actions_all_vehs, process_fn)
    return new_scored_actions_all_vehs


def add_reward_to_score(
    scored_actions_all_vehs,
    envt
):
    def add_reward(envt, action, score):
        reward = envt.get_reward(action)
        return (action, reward + score)
    process_fn = partial(add_reward, envt)

    new_scored_actions_all_vehs = _filter_actions(scored_actions_all_vehs, process_fn)
    return new_scored_actions_all_vehs

def add_driver_fairness_to_score(
    scored_actions_all_vehs,
    envt,
    vehs,
    discounted = False,
    plus = False,
):
    """
    Add driver fairness to the score of each action.
    Uses envt.alpha_d to select the fraction of fair vehicles.
    Uses envt.delta to scale the fairness bonus.
    Does not use the environment's copy of the driver_earnings

    if discounted, uses the vehice's discounted earnings
    """
    if envt.delta==0:
        return scored_actions_all_vehs
    if discounted:
        driver_earnings = np.array([v.discounted_value for v in vehs])
    else:
        driver_earnings = np.array([v.earning for v in vehs])

    num_fair_vehs = int(envt.alpha_d*envt.NUM_VEHS)
    if plus: num_fair_vehs=envt.NUM_VEHS
    fair_vehs = np.zeros(len(scored_actions_all_vehs))
    #Select worst-off alpha vehicles
    fair_vehs[np.argsort(driver_earnings)[:num_fair_vehs]] = 1

    if np.max(driver_earnings)==0 or num_fair_vehs==0:
        print("NOT ADDING DRIVER FAIRNESS")
        return scored_actions_all_vehs
    norm_earnings = driver_earnings/np.max(driver_earnings)
    avg_earning = np.mean(norm_earnings)

    new_scored_actions_all_vehs = []
    bonuses = []
    for i,scored_actions_per_veh in enumerate(scored_actions_all_vehs):
        new_scored_actions = []
        bonus = envt.delta * (avg_earning - norm_earnings[i])
        if plus: bonus = max([0,bonus])
        bonuses.append(bonus)
        for action, score in scored_actions_per_veh:
            # new_score = score+bonus
            new_score = score + len(action.requests)*bonus*fair_vehs[i] #New way
            new_scored_action = (action, new_score)
            new_scored_actions.append(new_scored_action)
        new_scored_actions_all_vehs.append(new_scored_actions)
    return new_scored_actions_all_vehs


def get_fscore(envt, req):
    frm,to = envt.zone_mapping[req.pickup], envt.zone_mapping[req.dropoff]
    if envt.fairtype=='src':
        fscore = envt.mean_source_sr - envt.source_sr[frm][0]
    else:
        fscore = envt.mean_pair_sr - envt.pair_sr[frm][to][0]
    return envt.beta*fscore

def add_fairness_to_score(scored_actions_all_vehs, requests, envt, fairness_potential=False):
    #If not adding fairness, return
    if envt.alpha==0 and envt.beta==0:
        return scored_actions_all_vehs
    # print(f"Adding rider fairness: {envt.fairtype},{envt.alpha},{envt.beta}")

    #Fairness bonus calculated based on service rate
    def pair_fair(frm, to):
        fscore = envt.mean_pair_sr - envt.pair_sr[frm][to][0]
        return fscore
    def src_fair(frm, to):
        fscore = envt.mean_source_sr - envt.source_sr[frm][0]
        return fscore

    fair_type = envt.fairtype
    onlyfair = envt.value_function=='xaveh'
    if fair_type == 'pair' or fair_type=='':
        fair_fn = pair_fair
    elif fair_type == 'src':
        fair_fn = src_fair

    def get_fscore(req):
        frm,to = envt.zone_mapping[req.pickup], envt.zone_mapping[req.dropoff]
        fscore = fair_fn(frm,to)
        if fairness_potential:
            fscore+= envt.source_sr[frm][0] - envt.source_sr[to][0]
        return envt.beta*fscore
    
    #Fair vehicles
    f_target = envt.fairness_target
    num_fair_vehs = int(envt.alpha*envt.NUM_VEHS)
    if f_target=='veh':
        fair_vehs = np.zeros(len(scored_actions_all_vehs))
        #Select first alpha vehicles
        for i in range(num_fair_vehs):
            fair_vehs[i]=1
    else:
        fair_vehs = np.ones(len(scored_actions_all_vehs)) #all fair
    
    #Fair requests
    if f_target in ['areq','+req']:
        fair_reqs = set()
        scored_requests = []
        for req in requests:
            scored_requests.append([req, get_fscore(req)])
        sorted_reqs = sorted(scored_requests, key=lambda x:x[1], reverse = True)
        if f_target == 'areq':
            num_fair_reqs = int(envt.alpha*len(requests))
            for i,(req,score) in enumerate(sorted_reqs):
                if i<num_fair_reqs:
                    fair_reqs.add(req) 
        elif f_target=='+req':
            for i,(req,score) in enumerate(sorted_reqs):
                if score>=0:
                    fair_reqs.add(req) 
        # print("fairreqs=", len(fair_reqs), len(sorted_reqs))
    else:
        fair_reqs = set(requests)

    #Calculate new scores
    new_scored_actions_all_vehs= [] #: List[List[Tuple[Action, float]]] 
    for i,scored_actions_per_veh in enumerate(scored_actions_all_vehs):
        new_scored_actions = []
        for action, score in scored_actions_per_veh:
            bonus = 0
            for req in action.requests:
                if req in fair_reqs:
                    bonus+=get_fscore(req)
            if onlyfair:
                new_score = score + fair_vehs[i]*(bonus - score)
            else:
                new_score = score + fair_vehs[i]*bonus #only for the first alpha fraction of vehicles
            new_scored_action = (action, new_score)
            new_scored_actions.append(new_scored_action)

        new_scored_actions_all_vehs.append(new_scored_actions)
    
    return new_scored_actions_all_vehs


def get_flattened_scores(scored_actions_all_vehs):
    return [score for scored_actions in scored_actions_all_vehs for _, score in scored_actions]


def dict2mat(dic, key_order=None, select_index=None):
    #Converts a nested dictionary into a 2D matrix. 
    #If dictionary is not nested, matrix is of shape (n,1)
    #Assumes same keys are used for both layers of dictionaries
    #select_index: if the elements in the nested dictionary are lists, specifies which item to keep. If None, keeps all
    mat = []
    # print("Here")
    if key_order is None:
        key_order = list(dic.keys())
    def sel_ind(item):
        if select_index is None:
            return item
        return item[select_index]

    for key in key_order:
        row = []
        if type(dic[key])!=dict:
            row.append(sel_ind(dic[key]))
        else:
            for key2 in key_order:
                row.append(sel_ind(dic[key][key2]))
        mat.append(row)
    # print(mat)
    return mat

def gini(x, return_mean=False):
    #calculates the gini index of distribution x
    if not len(x):
        if return_mean:
            return 0, 0
        else:
            return 0
    x = np.array(x)
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    avg = np.mean(x)
    gini =  diffsum / (len(x)**2 * avg)
    if return_mean:
        return gini, avg
    else:
        return gini

def pareto_frontier(Xs, Ys, ref=None, maxX = True, maxY = True):
    #The ref variable is an exogenous variable which moves us along the pareto front. 
    if not len(ref):
        ref = [None for i in range(len(Xs))]
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i], ref[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    ref_front = [pair[2] for pair in p_front]
    return p_frontX, p_frontY, ref_front

def get_driver_earning(action):
    earning = 0
    for req in action.requests:
        earning+=req.price
    return earning

def change_variance_rider(envt,action):
    """
    Raman et.al: Change in variance of the rider's utility
    Compute variance before and after the action, return the difference
    
    No discounting
    Only for pair fairness
    #will not work with training unless srs are loaded
    """
    #Flatten envt.pair_sr
    srs = {}
    for k,g in envt.pair_sr.items():
        for k2, sr_info in g.items():
            srs[(k,k2)] = sr_info
    srs_post = deepcopy(srs)

    for req in action.requests:
        pickup = envt.zone_mapping[req.pickup]
        dropoff = envt.zone_mapping[req.dropoff]
        old_sr, served, total = srs_post[(pickup, dropoff)]
        new_sr = (served+1)/(total+1)
        srs_post[(pickup, dropoff)] = (new_sr, served+1, total+1)
        
    pair_srs_prev = [sr_info[0] for sr_info in srs.values()]
    pair_srs_post = [sr_info[0] for sr_info in srs_post.values()]
    var_prev = np.var(pair_srs_prev)
    var_post = np.var(pair_srs_post)
    return var_post - var_prev

def change_variance_driver(envt, action):
    """
    Raman et.al: Change in variance of the driver's utility
    Compute variance before and after the action, return the difference
    
    No discounting
    #will not work with training unless earnings are loaded
    """
    var_prev = np.var(envt.driver_earnings)
    new_earnings = deepcopy(envt.driver_earnings)
    
    # calculate new earnings
    earning = get_driver_earning(action)
    new_earnings[action.veh_id] += earning

    var_post = np.var(new_earnings)
    return var_post - var_prev

def unroll_dict(d):
    #Assume 2 nested layers. Specifically for envt.pair_sr
    vs = []
    for k,v in d.items():
        v_sub = []
        for k2, v2 in v.items():
            v_sub.append(v2[0])
        vs.extend(v_sub)
    return vs

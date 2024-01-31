from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from eoscsp import EOSCSP, Observation, Request, Satellite
from greedy import first_slot, greedy_eoscsp_solver
from utils import generate_random_esop_instance


def bid(request: Request, R: Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]]) -> Optional[
    Tuple[float, Tuple[Observation, float]]]:
    # This function calculates the bid for a request based on the current plan
    # The bid is a tuple of the form (bid_value, winning_observation)
    o_sorted = sorted(request.theta, key=lambda obs: obs.t_start)
    for o in o_sorted:
        t = first_slot(o, R)
        if t is not None:
            return o.rho, (o, t[1])
    return 0, (None, -1)


def try_add(M, sig_u):
    # Attempt to add a new observation to the plan
    new_obs, new_start = sig_u
    idx = []
    removed = []
    reward = 0
    
    for i, (o, (s, start)) in enumerate(M):
        if o.s != new_obs.s:
            continue
        
        if (start <= new_start <= start + o.delta) or (new_start <= start <= new_start + new_obs.delta):
            idx.append(i)
            reward += o.rho
            
            if reward >= new_obs.rho:
                return False, []
            
            if len(idx) == 2:
                break
    
    for i in reversed(idx):
        removed.append(M[i][0].request.id)
        M.pop(i)
    
    return True, removed


def psi_solver(p: EOSCSP):
    """
    PSI Solver for the EOSCSP.
    :param p: An instance of EOSCSP.
    :return: A mapping from each observation to (satellite, start_time).
    """
    plans = []
    
    # Non-exclusive requests sorted by end time
    not_exclusive_requests = [r for r in p.requests if not r.u.exclusive_times]
    not_exclusive_requests.sort(key=lambda r: r.t_end)
    
    B_u = []  # List to store bids
    sig_u = []  # List to store signatures (plans) corresponding to each bid
    
    # Iterate over exclusive users and calculate bids for non-exclusive requests
    for user in p.users:
        if user.exclusive_times:
            sub_p = EOSCSP(satellites=p.satellites, users=[user], requests=[r for r in p.requests if r.u == user],
                           observations=[o for o in p.observations if o.u == user])
            _, r = greedy_eoscsp_solver(sub_p)
            
            plans.extend([x for value in r.values() for x in value])
            
            bids = [bid(req, deepcopy(r)) for req in not_exclusive_requests]
            B_u.append([b[0] for b in bids if b])
            sig_u.append([b[1] for b in bids if b])
    
    max_bid = np.max(B_u, axis=0)
    b_t = np.transpose(B_u)
    processed_requests = set()
    
    # Determine winning bids and update plans
    for i, req in enumerate(not_exclusive_requests):
        if max_bid[i] <= 0:
            continue
        w = np.where(b_t[i] == max_bid[i])[0][0]
        
        added, removed = try_add(plans, sig_u[w][i])
        if added:
            sat = sig_u[w][i][0].s
            plans.append((sig_u[w][i][0], (sat, sig_u[w][i][1])))
            processed_requests.add(req.id)
            for r_id in removed:
                processed_requests.remove(r_id)
    
    # Sort plans for execution order
    plans.sort(key=lambda x: (x[1][1]))
    R_ex = {sat.id: [] for sat in p.satellites}
    for obs, (start, end) in plans:
        R_ex[obs.s.id].append((obs, (start, end)))
    
    # Schedule remaining non-exclusive requests
    remaining_requests = [req for req in not_exclusive_requests if req.id not in processed_requests]
    obs = [obs for req in remaining_requests for obs in req.theta]
    p_u0 = EOSCSP(satellites=p.satellites, users=[p.users[0]], requests=remaining_requests, observations=obs)
    _, r = greedy_eoscsp_solver(p_u0, R_ex)
    M = [x for value in r.values() for x in value]
    
    # Calculate total reward
    total_reward = sum([o.rho for o, _ in M])
    print("Reward of psi: ", total_reward)
    
    # convert to dictionary
    final_solution = {}
    for observation, (satellite, start_time) in M:
        final_solution[observation.id] = (satellite, start_time)
    
    return final_solution


def ssi_solver(p: EOSCSP):
    plans = []
    
    # Non-exclusive requests sorted by end time
    not_exclusive_requests = [r for r in p.requests if not r.u.exclusive_times]
    not_exclusive_requests.sort(key=lambda r: r.t_end)
    
    B_u = []  # List to store bids
    sig_u = []  # List to store signatures (plans) corresponding to each bid
    
    R_ex = {sat.id: None for sat in p.satellites}
    # Iterate over exclusive users and calculate bids for non-exclusive requests
    for user in p.users:
        if user.exclusive_times:
            sub_p = EOSCSP(satellites=p.satellites, users=[user], requests=[r for r in p.requests if r.u == user],
                           observations=[o for o in p.observations if o.u == user])
            _, r = greedy_eoscsp_solver(sub_p)
            
            plans.extend([x for value in r.values() for x in value])
            R_ex[user.id] = r
    
    processed_requests = set()
    for i in range(len(not_exclusive_requests)):
        # bids
        bids = [bid(not_exclusive_requests[i], R_ex[user.id]) for user in p.users[1:]]
        B_u, sig_u = [b[0] for b in bids], [b[1] for b in bids]
        max_bid = np.max(B_u)
        
        # pass if no exclusive user scheduled this request
        if max_bid <= 0:
            continue
        
        w = np.where(B_u == max_bid)[0]
        w = np.random.choice(w)  # randomly choose one if there are multiple max bids
        
        added, removed = try_add(plans, sig_u[w])
        if added:
            sat = sig_u[w][0].s
            plans.append((sig_u[w][0], (sat, sig_u[w][1])))
            processed_requests.add(not_exclusive_requests[i].id)
            for i in removed:
                processed_requests.remove(i)
    
    plans.sort(key=lambda x: (x[1][1]))
    R_ex = {sat.id: [] for sat in p.satellites}
    for obs, (start, end) in plans:
        R_ex[obs.s.id].append((obs, (start, end)))
    
    # add non exclusive requests
    remaining_requests = [req for req in not_exclusive_requests if req.id not in processed_requests]
    obs = [obs for req in remaining_requests for obs in req.theta]
    p_u0 = EOSCSP(satellites=p.satellites, users=[p.users[0]], requests=remaining_requests, observations=obs)
    
    _, r = greedy_eoscsp_solver(p_u0, R_ex)
    M = [x for value in r.values() for x in value]
    
    # Calculate total reward
    total_reward = sum([o.rho for o, _ in M])
    print("Reward of ssi: ", total_reward)
    
    final_solution = {}
    for observation, (satellite, start_time) in M:
        final_solution[observation.id] = (satellite, start_time)
    
    return final_solution


if __name__ == '__main__':
    eoscsp = generate_random_esop_instance(4, 3, 10)
    schedule = ssi_solver(eoscsp)
    eoscsp.plot_schedule(schedule)

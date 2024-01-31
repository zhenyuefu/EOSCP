import json
import subprocess
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Set, Tuple

import yaml

from eoscsp import EOSCSP, Observation, Request, Satellite
from greedy import first_slot, greedy_eoscsp_solver
from utils import generate_random_esop_instance


def s_dcop(p: EOSCSP):
    user_solutions = {user.id: [] for user in p.users}
    # dict[userid] = dict[satid] = [(obs, (sat, start_time))
    rs = dict()
    processed_requests = set()
    
    # slove P[u] for each exclusive user
    for user in p.users:
        if user.exclusive_times:
            # create a sub problem P[u] only has request and observations of user u
            sub_p = EOSCSP(satellites=p.satellites, users=p.users, requests=[r for r in p.requests if r.u == user],
                           observations=[o for o in p.observations if o.u == user])
            plans, r,_ = greedy_eoscsp_solver(sub_p)
            user_solution = [x for value in r.values() for x in value]
            user_solutions[user.id] = user_solution
            rs[user.id] = r
            for obs, (sat, start_time) in user_solution:
                processed_requests.add(obs.request.id)
    
    # get the remaining requests
    remaining_requests = [r for r in p.requests if r.id not in processed_requests]
    sort_r = sorted(remaining_requests, key=lambda r: (r.u.p, r.t_start))
    
    R_ex = defaultdict(list)
    
    for request in sort_r:
        generate_dcop_yaml(p, request, user_solutions, rs)
        
        dcop_solution = solve_dcop()
        
        # print(dcop_solution)
        
        for varname, v in dcop_solution.items():
            if v == 1:
                userid, satid, obsid = varname.split('_')[1:]
                user_solutions[int(userid)].append(
                    (p.observations[int(obsid)], (p.satellites[int(satid)], p.observations[int(obsid)].t_start)))
                R_ex[int(satid)].append((p.observations[int(obsid)], (p.satellites[int(satid)], p.observations[int(obsid)].t_start)))
    # slove P[u_0] for non-exclusive user
    remaining_requests = [req for req in p.requests if req.id not in processed_requests]
    obs = [obs for req in remaining_requests for obs in req.theta]
    p_u0 = EOSCSP(satellites=p.satellites, users=[p.users[0]], requests=remaining_requests, observations=obs)
    _, r,_ = greedy_eoscsp_solver(p_u0, R_ex)
    M = [x for value in r.values() for x in value]
    total_reward = sum([o.rho for o, _ in M])
    print("Reward of sdcop: ", total_reward)
    final_solution = {}
    for observation, (satellite, start_time) in M:
        final_solution[observation.id] = (satellite, start_time)
    return final_solution, total_reward


def generate_dcop_yaml(p: EOSCSP,
                       request: Request,
                       user_solutions: Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]],
                       rs: Dict[int, Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]]]):
    dcop_data = {'name': 'EOSCSP', 'objective': 'max', 'domains': {'binary_decision': {'values': [0, 1]}}, 'variables': {}, 'agents': {},
                 'constraints': {}}
    
    observations = request.theta
    agents = set()
    for o in observations:
        t_start = o.t_start
        t_end = o.t_end
        for user in p.users:
            if user.exclusive_times:
                for sat, (start, end) in user.exclusive_times:
                    if sat.id == o.s.id:
                        if not (start >= t_end or end <= t_start):
                            agents.add((user.id, sat.id, o.id))
    
    # Generate agents based on exclusive users
    user_list = [userid for userid, satid, obsid in agents]
    user_list = set(user_list)
    agent_list = [f'u_{userid}' for userid in user_list]
    dcop_data['agents'] = agent_list
    
    # Generate variables and constraints for each request and observation
    variables_list = []
    obs_group = defaultdict(list)
    sat_group = defaultdict(list)
    for userid, satid, obsid in agents:
        var_name = f'x_{userid}_{satid}_{obsid}'
        variables_list.append(var_name)
        dcop_data['variables'][var_name] = {'domain': 'binary_decision'}
        obs_group[obsid].append(var_name)
        sat_group[satid].append(var_name)
    
    # Generate constraints for each observation
    for obsid, var_list in obs_group.items():
        constraint_name = f'one_obs_{obsid}'
        dcop_data['constraints'][constraint_name] = {'type': 'intention', 'function': f'100 if sum([{", ".join(var_list)}]) <= 1 else 0'}
    
    # Generate constraints for each satellite capacity
    for satid, var_list in sat_group.items():
        constraint_name = f'capacity_{satid}'
        remaine_capacity = calculate_capacity(p, satid, user_solutions)
        dcop_data['constraints'][constraint_name] = {'type': 'intention',
                                                     'function': f'100 if sum([{", ".join(var_list)}]) <= {remaine_capacity} else 0'}
    
    cost_function = build_cost_function(p, agents, rs)
    dcop_data['constraints']['cost'] = {'type': 'intention', 'function': cost_function}
    
    # distribute the variables to agents
    distribution = {}
    for variables in variables_list:
        userid = variables.split('_')[1]
        if f'u_{userid}' not in distribution:
            distribution[f'u_{userid}'] = []
        distribution[f'u_{userid}'].append(variables)
    
    with open('distribution.yaml', 'w') as file:
        yaml.dump({'distribution': distribution}, file, default_flow_style=False)
    
    # Serialize to YAML
    with open('dcop.yaml', 'w') as file:
        yaml.dump(dcop_data, file, default_flow_style=False)
    
    return dcop_data, distribution


def calculate_capacity(p: EOSCSP, satid: int, user_solutions: Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]]):
    # calculate the remaining capacity of a satellite
    capacity = p.satellites[satid].capacity
    for user in user_solutions:
        for obs, _ in user_solutions[user]:
            if obs.s.id == satid:
                capacity -= 1
    
    return capacity


def build_cost_function(p: EOSCSP,
                        agents: Set[Tuple[int, int, int]],
                        rs: Dict[int, Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]]]):
    cost_function = []
    for userid, satid, obsid in agents:
        var_name = f'x_{userid}_{satid}_{obsid}'
        reward = calculate_reward(p.observations[obsid], rs[userid])
        cost_function.append(f'{var_name} * {reward}')
    return f'sum([{", ".join(cost_function)}])'


def calculate_reward(o: Observation, r: Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]]):
    if first_slot(o, deepcopy(r)):
        return o.rho
    return 0


def solve_dcop() -> Dict:
    try:
        command = f"pydcop solve --algo dpop dcop.yaml -d distribution.yaml"
        process = subprocess.Popen(["/bin/bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        output = stdout.decode()
        result = json.loads(output)
        assignment = result.get("assignment", {})
        return assignment
    except:
        return False


def integrate_solutions(eoscsp, user_solutions):
    pass


if __name__ == '__main__':
    eoscsp = generate_random_esop_instance(4, 3, 8)
    schedule,reward = s_dcop(eoscsp)
    eoscsp.plot_schedule(schedule)

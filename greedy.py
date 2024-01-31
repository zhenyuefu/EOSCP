from typing import Dict, List, Optional, Tuple

from eoscsp import EOSCSP, Observation, Satellite
from utils import generate_random_esop_instance


def first_slot(observation: Observation, R: Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]]) -> Optional[
    Tuple[Satellite, float]]:
    s = observation.s
    if len(R[s.id]) < s.capacity:
        if not R[s.id]:  # R[s.id] is empty
            if observation.t_end >= observation.t_start + observation.delta:
                R[s.id].append((observation, (s, observation.t_start)))
                return s, observation.t_start
        else:
            for i in range(len(R[s.id]) + 1):
                t_start_prime = observation.t_start
                if i > 0:
                    obs_prev, (_, t_prev_start) = R[s.id][i - 1]
                    t_start_prime = max(observation.t_start, t_prev_start + obs_prev.delta + s.transition_time)
                if t_start_prime + observation.delta <= observation.t_end:
                    if i == len(R[s.id]):
                        t_upper = observation.t_end
                        t_end_prime = t_start_prime + observation.delta
                    else:
                        o_i, (s_i, t_next_start) = R[s.id][i]
                        t_upper = t_next_start
                        t_end_prime = t_start_prime + observation.delta + s.transition_time
                    if t_start_prime < t_end_prime <= t_upper:
                        R[s.id].insert(i, (observation, (s, t_start_prime)))
                        return s, t_start_prime
    return None


def greedy_eoscsp_solver(p: EOSCSP, r=None) -> Tuple[
    Dict[int, Tuple[Satellite, float]], Dict[int, List[Tuple[Observation, Tuple[Satellite, float]]]]]:
    # mapping from observation to (satellite, start_time)
    m = {}
    sorted_observations = sorted(p.observations, key=lambda obs: (obs.p, obs.t_start))
    # r[s.id] = [(o, (s, t_start))]
    if r is None:
        r = {s.id: [] for s in p.satellites}
    
    while sorted_observations:
        o = sorted_observations[0]  # Always work with the first element
        t = first_slot(o, r)
        if t is not None:
            m[o.id] = t
            # Remove the observation opportunities of the same request
            sorted_observations = [obs for obs in sorted_observations if obs.request != o.request]
        else:
            # Move to the next observation if no slot is found
            sorted_observations.pop(0)
    
    M = [x for value in r.values() for x in value]
    # Calculate total reward
    total_reward = sum([o.rho for o, _ in M])
    print("Reward of greedy: ", total_reward)
    return m, r


if __name__ == '__main__':
    eoscsp = generate_random_esop_instance(3, 2, 5)
    schedule, r = greedy_eoscsp_solver(eoscsp)
    eoscsp.plot_schedule(schedule)

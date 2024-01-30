from typing import List, Tuple, Optional, Dict

from eoscsp import EOSCSP, Observation, Satellite
from utils import generate_random_esop_instance


def first_slot(observation: Observation, R: Dict[int, List]) -> Optional[Tuple[Satellite, float]]:
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
                    _, (_, t_prev_end) = R[s.id][i - 1]
                    t_start_prime = max(observation.t_start, t_prev_end + s.transition_time)
                if t_start_prime + observation.delta <= observation.t_end:
                    if i == len(R[s.id]):
                        t_end_prime = t_start_prime + observation.delta
                    else:
                        _, (_, t_next_start) = R[s.id][i]
                        t_end_prime = min(t_next_start, t_start_prime + observation.delta)
                    if t_start_prime < t_end_prime:
                        R[s.id].insert(i, (observation, (s, t_start_prime)))
                        return s, t_start_prime
    return None


def greedy_eoscsp_solver(p: EOSCSP) -> Dict[int, Tuple[Satellite, float]]:
    # mapping from observation to (satellite, start_time)
    m = {}
    sorted_observations = sorted(p.observations, key=lambda obs: (obs.p, obs.t_start))
    request_dict = {s.id: [] for s in p.satellites}
    
    for o in sorted_observations:
        t = first_slot(o, request_dict)
        if t is not None:
            m[o.id] = t
            # Remove the observation opportunities of the same request
            sorted_observations = [obs for obs in sorted_observations if obs.request != o.request]
    return m


if __name__ == '__main__':
    eoscsp = generate_random_esop_instance(3, 2, 5)
    schedule = greedy_eoscsp_solver(eoscsp)
    eoscsp.plot_schedule(schedule)
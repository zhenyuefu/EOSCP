from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Patch

r_counter = count()
sat_counter = count()
u_counter = count()
obs_counter = count()

def reset_counters():
    global r_counter, sat_counter, u_counter, obs_counter
    r_counter = count()
    sat_counter = count()
    u_counter = count()
    obs_counter = count()

@dataclass
class Request:
    r"""
    A request is defined as a tuple with the following properties: :math:`r = (t^start, t^end, \delta, \rho, p, u, \theta)`.
    :param t_start: The start time of the request's valid time window, belonging to the set of real numbers R.
    :param t_end: The end time of the request's valid time window, belonging to the set of real numbers R.
    :param delta: The duration of the request, belonging to the set of real numbers R.
    :param reward: The reward offered if the request is fulfilled, belonging to the set of real numbers R.
    :param u: The identifier of the requester, belonging to the set of users U.
    :param theta: The list of observation opportunities to satisfy the request.
    """
    id: int = field(default_factory=lambda: next(r_counter), init=False)
    t_start: float
    t_end: float
    delta: float = field(init=False)
    reward: float
    u: 'User'
    theta: List['Observation'] = field(default_factory=list)
    
    def __post_init__(self):
        self.delta = self.t_end - self.t_start


@dataclass
class Satellite:
    r"""
    A satellite is defined as a tuple with the following properties: :math:`s = (t_start, t_end, K, τ)`.
    :param start_time: The start time of the satellite's orbital plan, belonging to the set of real numbers R.
    :param end_time: The end time of the satellite's orbital plan, belonging to the set of real numbers R.
    :param capacity: The maximum number of observations the satellite can make during its orbital plan, belonging to the set of positive
    integers N+.
    :param transition_time: The transition time for the satellite between two observations, belonging to the set of positive real numbers
    R+.

    """
    start_time: float
    end_time: float
    capacity: int
    transition_time: float
    id: int = field(default_factory=lambda: next(sat_counter))


@dataclass
class User:
    r"""
    A user is defined as a tuple with the following properties: :math:`u = (e_u, p_u)`.
    We denote U^ex (U^nex) as the set of users who have (or do not have) an exclusive segment of the orbit. There is only one user
    without an exclusive segment of the orbit, namely the central planner u_0, and there are no overlapping exclusive segments.
    :param exclusive_times: A set (which can be empty) of exclusive time window sets, defined as e_u={(s,(t^start, t^end))|s ∈ S,
    [t^start, t^end] ⊆ [t_s^start, t_s^end]}
                    ⊂ (S×(R×R)). Where s represents the satellite, [t^start, t^end] represents the time interval, and this interval is a
                    subset of the effective time window [t_s^start, t_s^end]
                    of satellite s.
    :param p: Priority, belonging to the set of positive integers N+ (the lower the number, the higher the priority, used for conflict
    resolution).
    """
    exclusive_times: List[Tuple[Satellite, Tuple[float, float]]]  # 独占时间窗口集合
    p: int = field(default=10)  # 优先级
    id: int = field(default_factory=lambda: next(u_counter))


@dataclass
class Observation:
    r"""
    An observation opportunity (or observation) is defined as a tuple with the following properties: :math:`o = (t^start, t^end, \delta,
    r, \rho, s, u, p)`.
    :param t_start: The start time of the observation's valid time window, belonging to the set of real numbers R.
    :param t_end: The end time of the observation's valid time window, belonging to the set of real numbers R.
    :param delta: The duration of the observation, belonging to the set of real numbers R, and :math:`delta_o = delta_r_o`.
    :param request: The request associated with the observation.
    :param rho: The reward for the observation, belonging to the set of real numbers R. It is determined by the request r_o and weather
    information.
    :param s: The satellite that can schedule this observation.
    :param u: The owner of the observation, belonging to the set of users U, and :math:`u = u_r_o`.
    :param p: The priority of the observation, belonging to the set of positive integers N+, and :math:`p = p_r_o`.
    The difference between request reward and observation reward arises from the fact that in real situations, weather conditions or the
    observation's angle of incidence may increase or decrease the basic reward for a given request.
    Therefore, our model can consider different rewards, but in this study, we focus on the case where the observation reward directly
    inherits from the request.
    """
    id: int = field(default_factory=lambda: next(obs_counter), init=False)
    i: int
    t_start: float
    t_end: float
    delta: float
    request: Request
    rho: float
    s: Satellite
    u: User
    p: int


@dataclass
class EOSCSP:
    r"""
    The Earth Observation Satellite Constellation Scheduling Problem is defined as a tuple :math:`P = (S, U, R, O)`, where S is the set
    of satellites, U is the set of users, R is the set of requests, and O is the set of observations that need to be scheduled to fulfill
    the requests in R.
    The goal of this problem is to efficiently schedule observations in the satellite constellation to satisfy the requests of users,
    while considering exclusive time windows and satellite capability constraints.
    :param satellites: The set of satellites S, containing multiple satellite objects.
    :param users: The set of users U, containing multiple user objects.
    :param requests: The set of requests R, containing multiple request objects.
    :param observations: The set of observations O, containing multiple observation objects.
    """
    satellites: List[Satellite]
    users: List[User]
    requests: List[Request]
    observations: List[Observation]
    
    def plot_schedule(self, s: Dict[int, Tuple[Satellite, float]] = None):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Generate distinct colors for each user
        colors = list(TABLEAU_COLORS)
        user_colors = {user.id: colors[i % len(colors)] for i, user in enumerate(self.users)}
        
        # Create a list for legend handles
        legend_handles = []
        for user in self.users:
            # Add to legend
            color = user_colors[user.id]
            handle = Patch(color=color, label=f'User {user.id}')
            legend_handles.append(handle)
        
        # Plotting satellites and their planning periods
        for i, satellite in enumerate(self.satellites):
            # ax.broken_barh([(satellite.start_time, satellite.end_time - satellite.start_time)], (i - 0.4, 0.8), facecolors='lightgrey')
            
            # Plotting exclusive orbit portions
            for user in self.users:
                color = user_colors[user.id]
                for excl_satellite, (excl_start, excl_end) in user.exclusive_times:
                    if excl_satellite == satellite:
                        ax.broken_barh([(excl_start, excl_end - excl_start)], (i - 0.4, 0.8), facecolors='none', edgecolor=color,
                                       hatch='//')
        
        # Height offset within each satellite's track for displaying observations
        observation_height_offset = 0.1
        max_observations_per_satellite = max(satellite.capacity for satellite in self.satellites)
        
        # Plotting observations
        for observation in self.observations:
            satellite_idx = self.satellites.index(observation.s)
            color = user_colors[observation.u.id]
            height = (satellite_idx - 0.4) + (observation.id % max_observations_per_satellite) * observation_height_offset
            
            if s and observation.id in s:
                # Use mapped satellite and start time if available
                mapped_satellite, mapped_start_time = s[observation.id]
                mapped_satellite_idx = self.satellites.index(mapped_satellite)
                height = (mapped_satellite_idx - 0.4) + (observation.id % max_observations_per_satellite) * observation_height_offset
                ax.broken_barh([(mapped_start_time, observation.delta)], (height, 0.1), facecolors=color)
            
            # observation window
            ax.broken_barh([(observation.t_start, observation.t_end - observation.t_start)], (height, 0.1), facecolors=color, alpha=0.2)
            
            # Annotate each observation
            mid_point = observation.t_start + (observation.t_end - observation.t_start) / 2
            ax.annotate(f'o_{{{observation.u.id},{observation.request.id},{observation.i}}}', xy=(mid_point, height), xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom', fontsize=8, color='black')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Satellites')
        ax.set_yticks(range(len(self.satellites)))
        ax.set_yticklabels([f'Satellite {s.id}' for s in self.satellites])
        ax.set_title('Satellite Observation Schedule')
        
        # Adding the legend to the plot
        ax.legend(handles=legend_handles, loc='upper right')
        
        plt.show()

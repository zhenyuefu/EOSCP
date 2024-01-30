from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Patch


@dataclass
class Request:
    r"""
    请求定义为具有以下属性的元组: :math:`r = (t^start, t^end, \delta, \rho, p, u, \theta)`。
    :param t_start: 请求的有效时间窗口开始时间，属于实数集 R。
    :param t_end: 请求的有效时间窗口结束时间，属于实数集 R。
    :param delta: 请求的持续时间，属于实数集 R。
    :param reward: 如果请求被满足，则提供的奖励，属于实数集 R。
    :param u: 请求者的标识，属于用户集 U。
    :param theta: 满足请求的观测机会列表。
    """
    id: int = field(default_factory=lambda counter=count(): next(counter), init=False)
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
    卫星定义为具有以下属性的元组：:math:`s = (t_start, t_end ,K, τ)`。
    :param start_time: 卫星轨道计划的开始时间，属于实数集 R。
    :param end_time: 卫星轨道计划的结束时间，属于实数集 R。
    :param capacity: 卫星在其轨道计划期间的最大观测数量，属于正整数集 N+。
    :param transition_time: 卫星在两个观测之间的转换时间，属于实数集 R+。
    """
    id: int = field(default_factory=lambda counter=count(): next(counter), init=False)
    start_time: float
    end_time: float
    capacity: int
    transition_time: float


@dataclass
class User:
    r"""
    用户定义为具有以下属性的元组: :math:`u = (e_u, p_u)`。
    我们记 U^ex（U^nex）为拥有（或不拥有）独占轨道部分的用户集合。只有一个用户没有独占的轨道部分，即中央计划者u_0，且不存在重叠的独占部分。
    :param exclusive_times: 一组（可能为空）的独占时间窗口集合，定义为 e_u={(s,(t^start, t^end))|s ∈ S,[t^start, t^end] ⊆ [t_s^start, t_s^end]}
                ⊂ (S×(R×R))。其中 s 代表卫星，[t^start, t^end] 代表时间区间，且该区间是卫星 s 的有效时间窗口 [t_s^start, t_s^end]
                的子集。
    :param p: 优先级，属于正整数集 N+（数值越低，优先级越高，用于解决冲突）。
    """
    id: int = field(default_factory=lambda counter=count(): next(counter), init=False)
    exclusive_times: List[Tuple[Satellite, Tuple[float, float]]]  # 独占时间窗口集合
    p: int = field(default=10)  # 优先级


@dataclass
class Observation:
    r"""
    观测机会（或观测）定义为具有以下属性的元组: :math:`o = (t^start, t^end, \delta, r, \rho, s, u, p)`。
    :param t_start: 观测的有效时间窗口开始时间, 属于实数集 R。
    :param t_end: 观测的有效时间窗口结束时间, 属于实数集 R。
    :param delta: 观测的持续时间, 属于实数集 R, 且 :math:`delta_o = delta_r_o`。
    :param request: 观测所关联的请求。
    :param rho: 观测的奖励, 属于实数集 R, 它是由请求 r_o 和天气信息综合决定的。
    :param s: 可以安排此观测的卫星。
    :param u: 观测的所有者, 属于用户集 U, 且 :math:`u = u_r_o`。
    :param p: 观测的优先级, 属于正整数集 N+, 且 :math:`p = p_r_o`。
    请求奖励与观测奖励之间的差异源于实际情况中, 天气条件或观测的入射角可能会增加或减少给定请求的基本奖励。
    因此, 我们的模型可以考虑不同的奖励, 但在这项研究中, 我们只关注观测奖励直接继承自请求的情况。
    """
    id: int = field(default_factory=lambda counter=count(): next(counter), init=False)
    i: int
    t_start: float
    t_end: float
    delta: float = field(init=False)
    request: Request
    rho: float
    s: Satellite
    u: User
    p: int
    
    def __post_init__(self):
        self.delta = self.t_end - self.t_start


@dataclass
class ESOP:
    r"""
    地球观测卫星星座调度问题定义为一个元组 :math:`P = (S, U, R, O)`，其中 S 是卫星集合，U 是用户集合，R 是请求集合，O 是需要调度以履行 R 中请求的观测集合。
    这个问题的目标是有效地调度卫星星座中的观测，以满足用户的请求，同时考虑独占时间窗口和卫星的能力限制。
    :param satellites: 卫星集合 S，包含多个卫星对象。
    :param users: 用户集合 U，包含多个用户对象。
    :param requests: 请求集合 R，包含多个请求对象。
    :param observations: 观测集合 O，包含多个观测对象。
    """
    satellites: List[Satellite]
    users: List[User]
    requests: List[Request]
    observations: List[Observation]
    
    def plot_schedule(self):
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
            bar = ax.broken_barh([(observation.t_start, observation.t_end - observation.t_start)], (height, 0.1), facecolors=color, alpha=0.2)
        
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


@dataclass
class Solution:
    r"""
    EOSCSP 的解决方案是一个映射 :math:`M={(o, t) | o \in O, t \in [t_o^{start}, t_o^{end}]}`，它为每个请求最多分配一个观测的开始时间，
    同时确保独占用户在其各自的独占窗口上有观测被安排。一个最优解是使总体奖励最大化的解决方案（安排的观测奖励之和）:math:`\argmax_M \sum_{(o, t) \in M} \rho_o`。
    :param schedule: 字典，将观测对象映射到其在独占窗口内的开始时间。
    """
    schedule: Dict[Observation, float]

import yaml
from pydcop.algorithms import *
from pydcop.dcop.yamldcop import load_dcop_from_file
from pydcop.infrastructure.run import solve

from eoscsp import EOSCSP, Observation, Satellite, User
from utils import generate_random_esop_instance


# 首先，定义一个函数来为DCOP创建代理
def create_agents(users: List[User]) -> List[str]:
    agents = [str(user.id) for user in users]
    return agents


# 定义一个函数来为DCOP创建变量
def create_variables(observations: List[Observation], users: List[User]) -> Dict[str, Dict]:
    variables = {}
    for user in users:
        for obs in observations:
            var_name = f"x_{user.id}_{obs.id}"
            variables[var_name] = {'domain': 'binary', 'initial_value': 0}
    return variables


# 定义一个函数来为DCOP创建约束
def create_constraints(satellites: List[Satellite], observations: List[Observation], users: List[User]) -> Dict[str, Dict]:
    constraints = {}
    # 添加每个请求最多一个观测的约束
    for obs in observations:
        constraint_name = f"request_max_one_{obs.request.id}"
        constraints[constraint_name] = {'type': 'intention',
                                        'function': f"sum([{', '.join(f'x_{user.id}_{obs.id}' for user in users)}]) <= 1"}
    # 添加卫星容量约束
    for sat in satellites:
        constraint_name = f"satellite_capacity_{sat.id}"
        constraints[constraint_name] = {'type': 'intention',
                                        'function': f"sum(["
                                                    f"{', '.join(f'x_{user.id}_{obs.id}' for user in users for obs in observations if obs.s == sat)}]) <= "
                                                    f"{sat.capacity}"}
    # 添加每个观测最多一个代理的约束
    for obs in observations:
        constraint_name = f"observation_single_agent_{obs.id}"
        constraints[constraint_name] = {'type': 'intention',
                                        'function': f"sum([{', '.join(f'x_{user.id}_{obs.id}' for user in users)}]) <= 1"}
    return constraints


# 构建DCOP实例
def build_DCOP(satellites: List[Satellite], users: List[User], observations: List[Observation]) -> Dict:
    agents = create_agents(users)
    variables = create_variables(observations, users)
    constraints = create_constraints(satellites, observations, users)
    
    dcop_instance = {'agents': agents, 'variables': variables, 'domains': {'binary': {'type': 'binary', 'values': [0, 1]}},
                     'constraints': constraints, 'objective': 'max', 'description': 'EOSCSP', 'name': 'EOSCSP'}
    return dcop_instance


# DCOP求解器
def solve_dcop(dcop_instance: Dict) -> Dict:
    # 将 DCOP 实例保存到 YAML 文件
    with open('dcop.yaml', 'w') as f:
        yaml.dump(dcop_instance, f)
    
    # 调用 pyDCOP 解算器
    dcop = load_dcop_from_file(['dcop.yaml'])
    distribution_strategy = 'oneagent'  # 'adhoc'
    assignment = solve(dcop, algo_def='dpop', distribution=distribution_strategy)
    
    return assignment


# 可视化解决方案
def visualize_solution(solution: Dict, eoscsp_instance: EOSCSP):
    # 调用EOSCSP实例的绘图方法
    eoscsp_instance.plot_schedule(solution)


# 在主函数中使用上述函数
if __name__ == '__main__':
    # 假设已经有了创建实例的函数
    eoscsp_instance = generate_random_esop_instance(3, 2, 5)
    dcop_instance = build_DCOP(eoscsp_instance.satellites, eoscsp_instance.users, eoscsp_instance.observations)
    
    # 解决DCOP问题
    solution = solve_dcop(dcop_instance)
    
    # 可视化解决方案
    visualize_solution(solution, eoscsp_instance)

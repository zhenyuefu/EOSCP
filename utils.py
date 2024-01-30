import random
from math import ceil

from esop import ESOP, Observation, Request, Satellite, User

OBSERVATION_PER_REQUEST = 2


# def generate_non_overlapping_times(start, end, num_windows):
#     times = sorted([random.uniform(start, end) for _ in range(num_windows * 2)])
#     return [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
def generate_non_overlapping_times(start, end, num_windows):
    minimum_duration = 1
    times = []
    
    current_start = start
    while len(times) < num_windows:
        t_start = random.uniform(current_start, end - minimum_duration*3)
        t_end = random.uniform(t_start + minimum_duration, end- minimum_duration)
        
        times.append((t_start, t_end))
        
        # 更新下一个窗口的最早开始时间，确保不重叠
        current_start = t_end
    
    return times


def generate_request_within_time_window(t_start, t_end, user):
    request_t_start = random.uniform(t_start, t_end - 1)
    request_t_end = random.uniform(request_t_start + 1, t_end)
    reward = random.uniform(10, 100)
    return Request(request_t_start, request_t_end, reward, user)


def add_theta(observations, request, satellite, user, i):
    observation = Observation(i, request.t_start, request.t_end, request, request.reward, satellite, user, user.p)
    observations.append(observation)
    request.theta.append(observation)


def generate_random_esop_instance(num_satellites, num_exclusive_users, num_requests):
    # start_time, end_time = random.randint(0, 8), random.randint(10, 24)
    start_time, end_time = 2, 20
    satellites = [Satellite(start_time=start_time, end_time=end_time, capacity=random.randint(4, 10), transition_time=0.2) for _ in
                  range(num_satellites)]
    
    users = [User(exclusive_times=[], p=random.randint(1, 10)) for _ in range(num_exclusive_users + 1)]  # Including central scheduler
    
    # Assign non-overlapping exclusive times to some users for each satellite
    for satellite in satellites:
        # make sure all exclusive users have at least one exclusive time
        if satellite.id == 0:
            exclusive_times = generate_non_overlapping_times(satellite.start_time, satellite.end_time, num_exclusive_users)
            user_shuffled = random.sample(users[1:], len(users[1:]))  # Exclude central scheduler
            
            for user, (excl_t_start, excl_t_end) in zip(user_shuffled, exclusive_times):
                user.exclusive_times.append((satellite, (excl_t_start, excl_t_end)))
        
        else:
            num_windows = random.randint(1, ceil(num_exclusive_users / 2))  # Random number of exclusive users per satellite
            exclusive_times = generate_non_overlapping_times(satellite.start_time, satellite.end_time, num_windows)
            
            for time_window in exclusive_times:
                user = random.choice(users[1:])  # Exclude central scheduler
                user.exclusive_times.append((satellite, time_window))
    
    requests = []
    observations = []
    for _ in range(num_requests):
        user = random.choice(users)
        # If the user has exclusive times, generate requests within those times
        if user.exclusive_times:
            satellite, (excl_t_start, excl_t_end) = random.choice(user.exclusive_times)
            request = generate_request_within_time_window(excl_t_start, excl_t_end, user)
            requests.append(request)
            
            # Generate observations for the request
            for i in range(OBSERVATION_PER_REQUEST):
                add_theta(observations, request, satellite, user, i)
        else:
            # Generate requests for central scheduler or users without exclusive times
            request = generate_request_within_time_window(start_time, end_time, user)
            requests.append(request)
            
            # Generate observations for the request
            for i in range(OBSERVATION_PER_REQUEST):
                obs_satellite = random.choice(satellites)
                add_theta(observations, request, obs_satellite, user, i)
    
    return ESOP(satellites=satellites, users=users, requests=requests, observations=observations)


if __name__ == '__main__':
    esop_instance = generate_random_esop_instance(3, 2, 5)
    esop_instance.plot_schedule()

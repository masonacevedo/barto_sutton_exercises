import numpy as np
from bandit import Bandit
import matplotlib.pyplot as plt

def sample_from_bandit(bandit, index):
    """
    Index is the slot arm you'd like to pull from.
    """
    distribution = bandit.distributions[index]
    return np.random.normal(loc=distribution[0], scale=distribution[1])


def get_best_arm(estimates):
    best_so_far = float('-inf')
    best_so_far_index = -1
    # best_so_far_num_samples = -1
    for index, (estimated_mean, num_samples) in enumerate(estimates):
        if estimated_mean > best_so_far:
            best_so_far = estimated_mean
            best_so_far_index = index
            # best_so_far_num_samples = num_samples
    return best_so_far_index


def run_experiment(distributions, EPSILON, num_pulls):
    
    
    bandit = Bandit(distributions)
    estimates = [[0.0, 0] for _ in range(0, len(distributions))]
    rewards = []
    for _ in range(0, num_pulls): 
        if np.random.uniform(0, 1) < EPSILON:
            index_to_choose = np.random.randint(0, len(distributions))
        else:
            index_to_choose = get_best_arm(estimates)
        sample = sample_from_bandit(bandit, index_to_choose)
        old_mean = estimates[index_to_choose][0]
        old_samples = estimates[index_to_choose][1]

        new_mean = old_mean + (sample - old_mean) / (old_samples + 1)
        new_samples = old_samples + 1
        estimates[index_to_choose] = [new_mean, new_samples]
        if len(rewards) == 0:
            rewards.append(sample)
        else:
            rewards.append(rewards[-1] + sample)
    print("EPSILON", EPSILON)
    for e in estimates:
        print(e)
    print("--------------------------------")
    return rewards

if __name__ == "__main__":
    means = list(np.random.normal(loc=0, scale=1, size=10))
    distributions = [(float(mean), 10.0) for mean in means]
    for d in distributions:
        print(d)
    print("--------------------------------")
    
    small_epsilon = run_experiment(distributions, EPSILON = 0.001, num_pulls = 100000)
    large_epsilon = run_experiment(distributions, EPSILON = 0.1, num_pulls = 100000)
    plt.plot(small_epsilon, label = "small epsilon")
    plt.plot(large_epsilon, label = "large epsilon")
    plt.legend()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

class arm:
    dist_class = "normal"
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.std_dev = np.sqrt(variance)
    def pull(self) -> float:
        # returns a random reward from the distribution
        return(self.mean + self.std_dev * np.random.standard_normal())

class multi_arm_bandit:
    dist_class = "normal"

    def __init__(self, num_arms, mean_reward_mean=10, mean_reward_var=4, arms_var=9):
        self.means = np.random.normal(mean_reward_mean, np.sqrt(mean_reward_var), num_arms)
        self.arms = []
        for i in range(num_arms):
            self.arms.append(arm(self.means[i], arms_var))
    def get_reward(self, action) -> float:
        return((self.arms[action]).pull())

if __name__ == "__main__":
    NUM_ARMS = 10
    bandit = multi_arm_bandit(NUM_ARMS)
    curr_mean = np.array([0] * NUM_ARMS)
    pulls = [0] * NUM_ARMS
    total_reward = 0
    regret = 0
    for i in range(10000):
        if(np.random.binomial(1, 0.1) == 1):
            action = np.random.randint(low=0, high=NUM_ARMS)
        else:
            action = np.argmax(curr_mean)
        # action = np.random.randint(low=0, high=NUM_ARMS)
        if action != bandit.means.argmax():
            regret += 1
        new_reward = bandit.get_reward(action)
        total_reward += new_reward
        curr_mean[action] = (pulls[action] * curr_mean[action] + new_reward) / (pulls[action] + 1)
        pulls[action] += 1
    print("Total reward in 10000 pulls:", total_reward)
    print("Optimal arm mean:", bandit.means.max())
    print("Regret:", regret)
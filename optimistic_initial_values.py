# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python/learn/v4/t/lecture/6399954?start=0

import numpy as np
import matplotlib.pyplot as plt
from sympy.functions.special.gamma_functions import uppergamma


class Bandit:
    def __init__(self, m, upper_limit):
        self.m = m # real mean
        self.mean = upper_limit
        self.N = 0
        
    def pull(self):
        return np.random.randn() + self.m
    
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N * x
        
def run_experiment(m1, m2, m3, eps, N):

    if eps == 0:
        bandits = [Bandit(m1,5), Bandit(m2,5), Bandit(m3,5)]
    else:
        bandits = [Bandit(m1,0), Bandit(m2,0), Bandit(m3,0)]
    
    data = np.empty(N)
    
    for i in range(N):
        # optimistic initial values if eps == 0
        if eps == 0:
            j = np.argmax([b.mean for b in bandits])
        else:
            p = np.random.random()
            if p < eps:
                j = np.random.choice(3)
            else:
                j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        
        # for the plot
        data[i] = x
        
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
        
    # # plot moving average ctr
    # plt.plot(cumulative_average)
    # plt.plot(np.ones(N)*m1)
    # plt.plot(np.ones(N)*m2)
    # plt.plot(np.ones(N)*m3)
    # plt.xscale('log')
    # plt.show()
    
    for b in bandits:
        print(b.mean)
        
    return cumulative_average

if __name__=='__main__':
    c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
    c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)
    c_optimistic = run_experiment(1.0, 2.0, 3.0, 0, 100000)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.plot(c_optimistic, label='Optimistic initial values')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    # # linear plot
    # plt.plot(c_1, label='eps = 0.1')
    # plt.plot(c_05, label='eps = 0.05')
    # plt.plot(c_01, label='eps = 0.01')
    # plt.legend()
    # plt.show()
    
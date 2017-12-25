# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python/learn/v4/t/lecture/6399954?start=0

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m):
        self.m = m # real mean
        self.mean = 0
        self.N = 0
        
    def pull(self):
        return np.random.randn() + self.m
    
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N * x
        
def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    
    data = np.empty(N)
    
    for i in range(N):
        #epsilon greedy, decaying
        p = np.random.random()
        if p < 1.0/(i+1):
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        
        # for the plot
        data[i] = x
        
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
        
    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    
    for b in bandits:
        print(b.mean)
        
    return cumulative_average

if __name__=='__main__':
    c_decay = run_experiment(1.0, 2.0, 3.0, 100000)

    # log scale plot
    plt.plot(c_decay, label='Decaying eps')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    # linear plot
    plt.plot(c_decay, label='Decaying eps')
    plt.legend()
    plt.show()
    
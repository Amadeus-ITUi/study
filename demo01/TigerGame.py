import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.k = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.k)
        self.regret = 0
        self.actions = []
        self.regrets = []
    
    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
        
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * bandit.k)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0,self.bandit.k)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k]+1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * bandit.k)
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 0.8
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0,self.bandit.k)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k]+1) * (r - self.estimates[k])
        return k

def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.title("%d-armed bandit"%solvers[0].bandit.k)
    plt.legend()
    plt.show()

np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)

# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [ EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons ]
# epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
# for solver in epsilon_greedy_solver_list:
#     solver.run(6000)
# plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(50000)
print(bandit_10_arm.probs)
print(decaying_epsilon_greedy_solver.estimates)
plot_results([decaying_epsilon_greedy_solver], ['Decaying Epsilon-Greedy'])
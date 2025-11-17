import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def show_beta(alpha,beta_param):
    xs = np.linspace(0,1,500)
    ys = beta.pdf(xs, alpha, beta_param)
    plt.plot(xs, ys, label=f'Beta({alpha},{beta_param})')
    # 抽几个样本示意
    samples = np.random.beta(alpha, beta_param, size=10)
    plt.scatter(samples, beta.pdf(samples, alpha, beta_param), marker='x')
    plt.legend()
    plt.show()

show_beta(1,1)   # 平的
show_beta(4,2)   # 中等，峰在 ~0.67
show_beta(61,41) # 很窄，峰在 ~0.598
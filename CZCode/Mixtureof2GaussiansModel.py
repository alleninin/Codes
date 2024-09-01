import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import random


k=2
##data1 = np.random.normal(loc=0, scale=.5, size=1000)
##data2 = np.random.normal(loc=5, scale=1.5, size=1000)
##data = np.concatenate([data1, data2])
##np.random.shuffle(data)
data = []

for i in range (k):
    data1 = np.random.normal(loc=random.random()*50, scale=random.random()*10, size=1000)
    data=np.concatenate([data,data1])
np.random.shuffle(data)

plt.hist(data, bins=50, density=True, alpha=0.6, color='g')

def em_gmm(data):
    mu1, mu2 = np.random.choice(data, 2)
    print(mu1)
    sigma1, sigma2 = np.std(data), np.std(data)
    pi = 0.5
    
    for i in range(150):
        r1 = pi * norm.pdf(data, mu1, sigma1)
        r2 = (1 - pi) * norm.pdf(data, mu2, sigma2)
        r = r1 + r2
        gamma1 = r1 / r
        gamma2 = r2 / r
        
        N1 = np.sum(gamma1)
        N2 = np.sum(gamma2)
        
        mu1 = np.sum(gamma1 * data) / N1
        mu2 = np.sum(gamma2 * data) / N2
        
        sigma1 = np.sqrt(np.sum(gamma1 * (data - mu1) ** 2) / N1)
        sigma2 = np.sqrt(np.sum(gamma2 * (data - mu2) ** 2) / N2)
        
        pi = N1 / len(data)
        
        
    return mu1, sigma1, mu2, sigma2, pi

mu1, sigma1, mu2, sigma2, pi = em_gmm(data)

x = np.linspace(min(data), max(data), 1000)
pdf1 = pi * norm.pdf(x, mu1, sigma1)
pdf2 = (1 - pi) * norm.pdf(x, mu2, sigma2)
plt.plot(x, pdf1, label='Component 1')
plt.plot(x, pdf2, label='Component 2')

print(pdf1[0])
plt.plot(x, pdf1 + pdf2, label='Combined')
plt.legend()
plt.show()

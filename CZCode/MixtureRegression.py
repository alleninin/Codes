import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import statistics
import random
np.random.seed(42)

n_samples = 1000
n_features = 1
k = 2 
X = np.hstack((np.ones((n_samples, 1)), np.random.rand(n_samples, n_features)))
print(X)
s=0.4
#spreading constant: smaller = more close
true_betas = [np.array([0, -6]), np.array([-5, 10])]  # [y-int,slope]
y = np.hstack([
    X[:n_samples//2] @ true_betas[0] + s*np.random.randn(n_samples//2),
    X[n_samples//2:] @ true_betas[1] + s*np.random.randn(n_samples//2)
])
print(y)
x_i=[]
for i in range (n_samples):
    x_i.append(X[i,1])

def em_gmm(X, y, k):
    n_samples, n_features = X.shape
    betalist = []
    pilist1 = []
    mulist = []
    sigmalist = []
    Nlist = []
    totalr = np.zeros(n_samples)
    rlist = [np.zeros(n_samples) for _ in range(k)]
    gammalist = [np.zeros(n_samples) for _ in range(k)]
    
    for i in range(k):
        Nlist.append(0)
        betalist.append(np.random.rand(n_features))
        mulist.append(np.random.rand())
        sigmalist.append(np.random.rand())
        pilist1.append(random.random())
    
    arranged = np.sum(pilist1)
    pilist = np.array(pilist1) / arranged

    for p in range(150):
        totalr.fill(0)
        
        for j in range(k):
            rlist[j] = pilist[j] * norm.pdf(y, X @ betalist[j], sigmalist[j])
        
        for j in range(k):
            totalr += rlist[j]
        for j in range(k):
            gammalist[j] = rlist[j] / totalr
        
        for j in range(k):
            Nlist[j] = np.sum(gammalist[j])
            W_j = np.diag(gammalist[j])

            XTWX = X.T @ W_j @ X
            XTWY = X.T @ W_j @ y
            betalist[j] = np.linalg.inv(XTWX) @ XTWY

            mulist[j] = np.sum(gammalist[j] * y) / Nlist[j]
            sigmalist[j] = np.sqrt(np.sum(gammalist[j] * (y - X @ betalist[j])**2) / Nlist[j])
        
        for j in range(k):
            pilist[j] = Nlist[j] / n_samples
        
    return betalist, sigmalist, pilist, mulist


betalist, sigmalist, pilist, mulist = em_gmm(X, y, k)

plt.figure(figsize=(10, 6))

plt.scatter(X[:, 1], y, color='grey', alpha=0.5)
plt.scatter(X[:n_samples//2, 1], y[:n_samples//2], color='blue', alpha=0.5, label='Data')
plt.scatter(X[n_samples//2:, 1], y[n_samples//2:], color='yellow', alpha=0.5, label='Data')

#plt.scatter(x_i, y, color='blue', alpha=0.5, label='Data')

x_values = np.linspace(0, 1, 100)
for j in range(k):
    y_values = betalist[j][0] + betalist[j][1] * x_values
    plt.plot(x_values, y_values, label=f'Component {j+1}', color='black')
print(betalist)
plt.xlabel('Predictor (X)')
plt.ylabel('Response (y)')
plt.legend()
plt.title('Mixture of Linear Regressions')
plt.show()

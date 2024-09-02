import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import random


k=3

data = []

for i in range (k):
    data1 = np.random.normal(loc=random.random()*50, scale=random.random()*10, size=1000)
    data=np.concatenate([data,data1])
np.random.shuffle(data)

plt.hist(data, bins=50, density=True, alpha=0.6, color='g')

def em_gmm(data):
    mulist = []
    pilist1 = []
    sigmalist = []
    Nlist = []
    totalr=[]
    rlist = []
    gammalist=[]
    for i in range(k):
        Nlist.append(0)
        rlist.append(0)
        gammalist.append(0)
    for i in range (k):
        mulist.append(np.random.choice(data,1))
        sigmalist.append(np.random.choice(data,1))
        pilist1.append(random.random())
    pi = 0.5
    arranged=np.sum(pilist1)
    pilist = pilist1/arranged
    for i in range(1000*k):
        totalr.append(0)
    for i in range(k):
        Nlist.append(0)
        rlist.append(0)
        #totalr.append(0)
        gammalist.append(0)
        
    for p in range(150):
        for i in range(1000*k):
            totalr[i] = (0)
        #issue
        for i in range(k):
            rlist[i] = pilist[i] * norm.pdf(data, mulist[i], sigmalist[i])
        #r1 = pilist[0] * norm.pdf(data, mulist[0], sigmalist[0])
        #print(r1[0])
        #r2 = pilist[1] * norm.pdf(data, mulist[1], sigmalist[1])
        
        r = rlist[0] + rlist[1]
##        print(rlist[0])
##        print(rlist[1])
##        print(r)
        for i in range (k):
            totalr +=rlist[i]
##        print(totalr)
##        print("")
            
        for i in range(k):
            gammalist[i] = (rlist[i] / totalr)
        #issue
        for i in range(k):
            Nlist[i] = (np.sum(gammalist[i]))
        
        for j in range(k):
            mulist[j] = np.sum(gammalist[j] * data) / Nlist[j]
            sigmalist[j] = np.sqrt(np.sum(gammalist[j] * (data - mulist[j]) ** 2) / Nlist[j])
        for j in range(k):
            pilist[j] = Nlist[j] / len(data)
        
    return mulist,sigmalist,pilist

mulist,sigmalist,pilist = em_gmm(data)

x = np.linspace(min(data), max(data), 1000)
pdfs = []



for i in range(k):
    pdfs.append(pilist[i] * norm.pdf(x, mulist[i], sigmalist[i]))
    plt.plot(x,pdfs[i],label='Component ' + str(i+1))


for i in range(1,k):
    pdfs[0]+=pdfs[i]

plt.plot(x,pdfs[0],label = 'combined')
plt.legend()
plt.show()

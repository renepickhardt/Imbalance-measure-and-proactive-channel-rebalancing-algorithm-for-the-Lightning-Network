import time
import sys
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

f = open("out/1574847007.523627_raw", "r")
healths = []
medians = []
means = []
lbcs = []
for line in f:
    vals = len(line.split("\t"))
    if vals == 4:
        node, health, median, mean = line.split("\t")
        healths.append(float(health))
        medians.append(float(median))
        means.append(float(mean))
    if vals == 2:
        node, local_balance_coefficient = line.split("\t")
        lbcs.append(float(local_balance_coefficient))

#a = np.random.normal(2, 1, size=100)
#b = np.random.normal(0, 1, size=100)
#lbcs = np.concatenate((a, b))
k2, p = stats.normaltest(lbcs)
print(k2, p)
alpha = 0.01
# FIXME: Do I have to do that measure in every step to see that it always made sense? I mean even initilly the local health measures looked normal distributed. but I guess if time is left this measure would be nice.
if p < alpha:
    print("does not come from a normal distribution")
else:
    print("does come from a normal distribution. Global health measure as average is reasonable")


plt.scatter(healths, medians)
plt.title("Comparing network health with the median size of routable payments")
plt.xlabel("network health score (low health is good)")
plt.ylabel("median possible payment amount accross all pairs of nodes")
plt.grid()
plt.show()

f = open("out/1574847007.523627_routability", "r")
flag = True
fails = []
mins = []
for line in f:
    if flag:
        flag = False
        print("ignore first line")
        continue
    values = [int(x) for x in line.split("\t")]
    numfails = 0
    min_amt = sys.maxsize
    for x in values:
        if x == 0:
            numfails += 1
        if x < min_amt:
            min_amt = x
    fails.append(float(numfails) / float(len(values)))
    mins.append(min_amt)

plt.scatter(healths, fails)
plt.grid()
plt.title(
    "Comparing The network health with the failure rate at first try routing attempt")
plt.xlabel("network health score (low health is good)")
plt.ylabel("failure rate (probability for the smallest payment to fail)")
plt.show()


plt.scatter(healths, mins)
plt.show()

"""
ideas:
difference median vs mean
median
percentage of failed payments

"""

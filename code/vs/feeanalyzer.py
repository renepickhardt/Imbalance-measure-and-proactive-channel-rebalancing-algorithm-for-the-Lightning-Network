import matplotlib.pyplot as plt
import numpy as np

f = open("better_balanced_directed_lightning_network_fees_3_5000_fees", "r")

fees = []
for l in f:
    fee = float(l[:-1].split("\t")[1])
    if abs(fee) > 100000:
        print(fee)
    else:
        fees.append(fee)
print(max(fees))

f = open("directed_lightning_network", "r")
for l in f:
    base = int(l[:-1].split("\t")[5])
    if base > 100000:
        print(base)

plt.hist(fees, bins=20)
plt.show()


#

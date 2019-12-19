import matplotlib.pyplot as plt
import numpy as np

f = open("finalResults/fullExperimentNonStrictRebalancing/better_balanced_directed_lightning_network_fees_3_5000_rebalancing_operations", "r")
amts = []
for operation in f:
    amt, circle = operation.split("\t")
    amt = int(amt)
    circle = circle.split(" ")
    amts.append(amt)

# plt.plot(amts)
# plt.show()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


avg = running_mean(amts, 50000)
plt.plot(avg)
# plt.yscale("log")
plt.grid()
plt.show()

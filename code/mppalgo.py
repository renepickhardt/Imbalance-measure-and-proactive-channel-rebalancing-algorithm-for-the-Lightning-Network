"""
7. Janary 2020 example code that demonstrates how the imbalance improvement measure should be applied for multi path payments on the lightning network. 

Author: Rene Pickhardt
"""


import numpy as np

# amount that is to be paid.
a = 0.7


def gini(x):
    # from https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    # by warren weckesser

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


# initial balance of channels
b = [0.1*i for i in range(10, 0, -1)]
print(gini(b), "initial imblance")

new_funds = sum(b) - a
# assuming all channels have capacity of 1 btc
cap = len(b)

nu = float(new_funds) / cap

print("new funds {} and new node balance coefficient {}".format(new_funds, nu))

ris = [1*(float(x)/1 - nu) for x in b]

real_ris = [x for x in ris if x > 0]
s = sum(real_ris)
if (a > sum(b)):
    print("not enough funds to conduct such a payment")
    exit()
payments = [a*x/s for x in real_ris]


print("\nConduct the following payments: ")
for i, payment in enumerate(payments):
    print("channel {} old balance: {:4.2f}, payment amount {:4.2f} new balance {:4.2f}".format(
        i, b[i], payment, b[i] - payment))
if i+1 < len(b):
    print("\n---- unchanged channels as they need more funds on our side ----\n")
    for j in range(i+1, len(b), 1):
        print("channel {} old balance: {:4.2f}, payment amount {:4.2f} new balance {:4.2f}".format(
            j, b[j], 0, b[j]))

print("total amount paid over several channels: ", sum(payments))
print("(was asked to send: {})".format(a))

new_b = [x for x in b]
for i, x in enumerate(payments):
    new_b[i] = b[i] - x

print("\nnew imbalance {:4.2f} and old imbalance {:4.2f}".format(
    gini(new_b), gini(b)))

import itertools
import random
f = open("names.tsv", "r")
names = []
for l in f:
    x = random.randint(1, 2)
    name = l[:-1].split("\t")[x]
    names.append(name)

nametuples = list(itertools.product(names, names))

items = random.sample(nametuples, 2000)
take = []
for item in items:
    if item[0] != item[1]:
        take.append(item)

f = open("network100", "w")
for item in take:
    f.write("{}\t{}\t{}\n".format(
        item[0], item[1], random.randint(1000, 10000)))
# f.write("\n")


f = open("names1000", "r")
names = [l[:-1] for l in f]
nametuples = list(itertools.product(names, names))

items = random.sample(nametuples, 20000)
take = []
for item in items:
    if item[0] != item[1]:
        take.append(item)

f = open("network1000", "w")
for item in take:
    f.write("{}\t{}\t{}\n".format(
        item[0], item[1], random.randint(1000, 10000)))
# f.write("\n")

f=open("names.tsv","r")
import random
import itertools
names = []
for l in f:
    x = random.randint(1,2)
    name = l[:-1].split("\t")[x]
    names.append(name)

nametuples = list(itertools.product(names,names))

items = random.sample(nametuples, 2000)
take = []
for item in items:
    if item[0] != item[1]:
        take.append(item)

f = open("network","w")
for item in take:
    f.write("{}\t{}\t{}\n".format(item[0],item[1],random.randint(1000,10000)))
f.write("\n")

    

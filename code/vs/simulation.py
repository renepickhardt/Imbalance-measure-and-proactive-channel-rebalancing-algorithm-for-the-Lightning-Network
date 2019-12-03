import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import sys
import numpy as np


class Network:

    def __add_channel_with_fees(self, fields):
        s, d, c, a, base, rate = fields
        self.G.add_edge(s, d, capacity=int(c), balance=int(a),
                        base=int(base), rate=int(rate))

    def __compute_rebalance_Graph(self):
        for u in self.G.nodes():
            total_funds = sum(self.G[u][v]["balance"] for v in self.G[u])
            total_capcity = sum(self.G[u][v]["capacity"] for v in self.G[u])
            self.nu[u] = float(total_funds)/total_capcity

    def __init__(self, file_name):
        self.G = nx.DiGraph()
        self.nu = {}
        f = open(file_name, "r")
        for line in f:
            fields = line[:-1].split("\t")
            if len(fields) == 6:
                self.__add_channel_with_fees(fields)
        print(len(self.G))
        self.__compute_rebalance_Graph()
        print("imported a graph with {} nodes and {} edges ".format(
            len(self.G), len(self.G.edges())/2))

    def compute_rebalance_directions(self):
        self.flow = nx.DiGraph()
        for u, v in self.G.edges():
            nu = self.nu[u]
            balance = self.G[u][v]["balance"]
            capacity = self.G[u][v]["capacity"]
            zeta = balance / capacity
            if zeta > nu:
                amt = int(capacity*(zeta - nu))
                self.flow.add_edge(u, v, liquidity=amt)

        # paths = dict(nx.all_simple_paths(self.flow, 6)

        """
        paths = dict(nx.all_pairs_shortest_path(self.flow, 8))
        self.paths = {}
        for u in self.flow:
            for v in self.flow[u]:
                if v in paths:
                    if u in paths[v]:
                        self.paths[u + ":" + v] = [paths[v][u]]
        print(len(self.paths), "amount of potential rebalance paths")
        """
        self.paths = {}
        for c, u in enumerate(self.flow):
            for v in self.flow[u]:
                key = u + ":" + v
                self.paths[key] = []
                for cnt, path in enumerate(nx.all_simple_paths(self.flow, v, u, 4)):
                    if cnt > 100:
                        #print("too many paths")
                        break
                    self.paths[key].append(path)
            if (c % 10) == 0:
                print(c)

    def __compute_rebalance_amount(self, circle):
        ptr = 1
        amt = sys.maxsize
        while ptr < len(circle):
            src = circle[ptr-1]
            dest = circle[ptr]
            ptr += 1
            tmp = self.flow[src][dest]["liquidity"]
            if tmp < amt:
                amt = tmp
        return amt

    def __update_channels(self, circle, amt):
        ptr = 1
        while ptr < len(circle):
            src = circle[ptr-1]
            dest = circle[ptr]
            ptr += 1
            self.G[src][dest]["balance"] -= amt
            self.G[dest][src]["balance"] += amt

    def __update_flow(self, circle, amt):
        ptr = 1
        while ptr < len(circle):
            src = circle[ptr-1]
            dest = circle[ptr]
            ptr += 1
            self.flow[src][dest]["liquidity"] -= amt

    def rebalance_circle(self, circle):
        assert circle[0] == circle[-1]
        assert len(circle) > 2
        amt = self.__compute_rebalance_amount(circle)
        if amt < 1:
            return
        #print(amt, "to be rebalanced")
        self.__update_flow(circle, amt)
        self.__update_channels(circle, amt)

    def make_one_step(self):
        #path = random.sample(list(self.paths.values()), 1)[0]
        cnt = 1
        for k, paths in self.paths.items():
            for path in paths:
                path.append(path[0])
                self.rebalance_circle(path)
                if cnt % 10000 == 0:
                    print(cnt)
                cnt += 1

    def gini(self, x):
        # FIXME: replace with a more efficient implementation
        mean_absolute_differences = np.abs(np.subtract.outer(x, x)).mean()
        # print(x)
        relative_absolute_mean = mean_absolute_differences/np.mean(x)
        # print(relative_absolute_mean)
        return 0.5 * relative_absolute_mean

    def health(self):
        ginis = []
        for u in self.G:
            zetas = []
            for v in self.G[u]:
                balance = self.G[u][v]["balance"]
                capacity = self.G[u][v]["capacity"]
                zeta = balance / capacity
                zetas.append(zeta)
            g = self.gini(zetas)
            ginis.append(g)
        return np.mean(ginis)


print("test")

n = Network("directed_lightning_network")
for i in range(200):
    start = time.time()
    n.compute_rebalance_directions()
    end = time.time()
    print(end-start, "time to compute directions")
    start = time.time()
    n.make_one_step()
    end = time.time()
    print(end-start, "time to rebalance the circles")
    start = time.time()
    print(n.health(), "imbalance (average of gini coefficients)")
    end = time.time()
    print(end-start, "time to compute the ginis")
    break


raw_data = open("better_balanced_network", "w")
for node in n.G:
    # raw_data.write(node)
    for adj in n.G[node]:
        capacity = n.G[node][adj]["capacity"]
        balance = n.G[node][adj]["balance"]

        raw_data.write("{}\t{}\t{}\t{}\t{}\n".format(
            node, adj, capacity, balance, balance/capacity))

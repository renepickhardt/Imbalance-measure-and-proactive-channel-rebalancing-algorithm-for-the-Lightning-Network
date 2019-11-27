import time
import sys
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

balance = "balance"
capacity = "capacity"


class Network:
    G = None
    shortest_paths = []

    def __init__(self):
        self.G = nx.DiGraph()

    def __initialize_balances(self):
        to_add = []
        for u in self.G:
            for v in self.G[u]:
                if (v, u) not in self.G:
                    c = self.G[u][v][capacity]
                    to_add.append((v, u, c))
        for v, u, c in to_add:
            self.G.add_edge(v, u, capacity=c, balance=0)

    def __all_pair_shortest_paths(self):
        res = nx.all_pairs_shortest_path(self.G)
        m = 0
        d = {}
        self.shortest_paths = []
        for p in res:
            for k, v in p[1].items():
                l = len(v)
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
                if l > m:
                    m = l
                if l == 1:
                    continue
                self.shortest_paths.append(v)
        print("diameter", m)
        print("distribution of shortest path length", d)

    def __beta(self, u, v):
        return float(self.G[u][v][balance])/float(self.G[u][v][capacity])

    def __get_outbound_channels(self, node):
        gamma = self.compute_gamma(node)
        return [n for n in self.G[node] if self.__beta(node, n) > gamma]

    def __get_inbound_channels(self, node):
        gamma = self.compute_gamma(node)
        return [n for n in self.G[node] if self.__beta(node, n) < gamma]

    def __find_foaf_circles(self, u, v):
        tmpG = nx.DiGraph()
        out_nodes = self.__get_outbound_channels(v)
        for x in out_nodes:
            tmpG.add_edge(v, x)
        in_nodes = self.__get_inbound_channels(u)
        #print(in_nodes, u, v, out_nodes)
        for x in in_nodes:
            tmpG.add_edge(x, u)
            foaf_in = self.__get_inbound_channels(x)
            for y in foaf_in:
                tmpG.add_edge(y, x)
        paths = []
        if u in tmpG and v in tmpG:
            try:
                #path = nx.dijkstra_path(tmpG, v, u)
                paths = nx.all_simple_paths(tmpG, v, u, 5)
            except:  # probably no path / circle exist
                pass
        return paths

    def __rebalance_circle(self, circle, amt):
        for i in range(len(circle)-1):
            f = circle[i]
            t = circle[i+1]
            self.G[f][t][balance] -= amt
            self.G[t][f][balance] += amt

    def __compute_rebalance_amt(self, circle):
        """
        cannot be done in the real lightning network like this

        since we do a simulation it is fine to compute this sloppy
        """
        amt = sys.maxsize
        for i in range(len(circle)-1):
            f = circle[i]
            t = circle[i+1]
            #gamma = np.mean(self.compute_betas(f))
            gamma = self.compute_gamma(f)
            beta = self.__beta(f, t)
            tmp = int((beta - gamma) * self.G[f][t][capacity])
            if tmp < amt:
                amt = tmp

        return amt

    def __compute_forward_amt(self, circle):
        amt = sys.maxsize
        # if len(circle) < 3:
        #    return 0
        # print(circle)
        for i in range(len(circle)-1):
            f = circle[i]
            t = circle[i+1]
            tmp = self.G[f][t][balance]
        #    print(f, t, tmp)
            if tmp < amt:
                amt = tmp
        # print(amt)
        """if amt == 0:
            for n in circle:
                self.make_node_healthier(n)

            amt = sys.maxsize
            if len(circle) < 3:
                return 0
            print(circle)
            for i in range(len(circle)-1):
                f = circle[i]
                t = circle[i+1]
                tmp = self.G[f][t][balance]
                print(f, t, tmp)
                if tmp < amt:
                    amt = tmp
                print(amt)
            exit()"""
        return amt

    def load_network(self):
        f = open("network", "r")
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            f, t, c = line.split("\t")
            e = (t, f)
            # FIXME: make sure the graph generator does not include back and forth channels
            if self.G.has_edge(*e):
                continue
            self.G.add_edge(f, t, capacity=int(c), balance=int(c))
        self.__initialize_balances()
        self.__all_pair_shortest_paths()

    def compute_gamma(self, node):
        tc = 0
        tb = 0
        try:
            for n in self.G[node]:
                tc += self.G[node][n][capacity]
                tb += self.G[node][n][balance]
        except:
            print(node, self.G)
        return float(tb)/float(tc)

    def beta(self, u, v):
        return self.__beta(u, v)

    def compute_betas(self, node):
        return [self.__beta(node, n) for n in self.G[node]]

    def gini(self, x):
        # FIXME: replace with a more efficient implementation
        mean_absolute_differences = np.abs(np.subtract.outer(x, x)).mean()
        relative_absolute_mean = mean_absolute_differences/np.mean(x)
        return 0.5 * relative_absolute_mean

    def total_health(self):
        return np.mean([self.gini(self.compute_betas(n)) for n in self.G])

    def routability(self):
        amts = []
        for path in self.shortest_paths:
            amt = self.__compute_forward_amt(path)
            # if amt < 1:
            #    amt = 0
            amts.append(amt)
        # plt.hist(amts)
        # plt.show()
        # print(amts)
        return np.median(amts), np.mean(amts), amts

    def make_node_healthier(self, node):
        gamma = self.compute_gamma(node)
        candidate = None
        max_diff = 0
        for n in self.G[node]:
            beta = self.__beta(node, n)
            diff = beta - gamma
            if diff > max_diff:
                max_diff = diff
                candidate = n
        if candidate is None:
            #print("Node {} has no candidates for rebalancing right now".format(node))
            # print(self.G[node])
            return -1
        # FIXME: check if we made sure that the inbound channel has a smaller beta_value than gamma / gamma
        circles = self.__find_foaf_circles(node, candidate)
        rebalance_options = []
        for circle in circles:
            c = [node]
            c.extend(circle)
            amt = self.__compute_rebalance_amt(c)
            if len(c) <= 3 or amt < 1:
                continue
            rebalance_options.append((c, amt))
        if len(rebalance_options) < 1:
            return -1
        k = int(len(rebalance_options)/2) + 1
        k = 1
        total_amt = 0
        for option in random.sample(rebalance_options, k):
            circle = option[0]
            amt = option[1]
            tmp_amt = max(1, int(amt/k))
            total_amt += tmp_amt
            self.__rebalance_circle(circle, tmp_amt)
        return total_amt
        #print(node, circle)


n = Network()
n.load_network()
# n.G.add_edge("B","A",capacity=100,balance=0)

run = "out/" + str(time.time())
raw_data = open(run+"_raw", "w")
routability_data = open(run+"_routability", "w")

nodes = list(n.G)
nodes_set = set(nodes)

ginis = [n.gini(n.compute_betas(node)) for node in nodes]

plt.hist(ginis)
plt.title("Distribution of Ginicoefficients before simulation")
plt.ylabel("frequency")
plt.xlabel("local ginicoefficient")
plt.grid()
plt.savefig(run + "_ginicoefficients_start.png")
# plt.show()
plt.close()


health = n.total_health()
median, mean, amts = n.routability()
routability_data.write("\t".join(str(amt) for amt in amts)+"\n")
health_ts = [health]
for step in range(25000):
    if len(nodes) == 0:
        break
    node = random.choice(nodes)
    rebalance_amt = n.make_node_healthier(node)

    if rebalance_amt < 0:
        nodes_set.remove(node)
        nodes = list(nodes_set)
    else:
        nodes = list(n.G)
        nodes_set = set(nodes)

    health = n.total_health()
    median, mean, amts = n.routability()
    print(median, mean, step, rebalance_amt, node,  health)
    raw_data.write("{}\t{}\t{}\t{}\n".format(node, health, median, mean))
    routability_data.write("\t".join(str(amt) for amt in amts)+"\n")
    health_ts.append(health)
routability_data.flush()
routability_data.close()
print(step, "steps used for rebalancing")

raw_data.write("\n")
for node in n.G:
    # raw_data.write(node)
    for adj in n.G[node]:
        raw_data.write("{}\t{}\t{}\t{}\t{}\n".format(
            node, adj, n.G[node][adj][capacity], n.G[node][adj][balance], n.beta(node, adj)))

raw_data.write("\n")
ginis = []
nodes = list(n.G)
for node in nodes:
    gini = n.gini(n.compute_betas(node))
    ginis.append(gini)
    raw_data.write("{}\t{}\n".format(node, gini))
raw_data.flush()
raw_data.close()


plt.plot(health_ts)
plt.title("Network health measured as evenly distributed liquidity")
plt.ylabel("average Ginicoefficient of local chanal balance coefficients")
plt.xlabel("simulation steps")
plt.grid()
plt.savefig(run+"_figure.png")
# plt.show()
plt.close()

plt.hist(ginis)
plt.title("Distribution of Ginicoefficients after simulation")
plt.ylabel("frequency")
plt.xlabel("local ginicoefficient")
plt.grid()
plt.savefig(run + "_ginicoefficients_end.png")
# plt.show()
plt.close()


"""
for node in n.G:
    betas = n.compute_betas(node)
    print(node, n.G[node], n.compute_gamma(node), betas, n.gini(betas))
    n.make_node_healthier(node)
print(n.total_health())

for node in n.G:
    betas = n.compute_betas(node)
    print(node, n.G[node], n.compute_gamma(node), betas, n.gini(betas))
"""

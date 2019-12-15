import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import sys
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dataset = "directed_lightning_network"
recompute_all_pair_shortest_paths = False
experiment_name = "finalResults/fullExperimentNonStrictRebalancing/better_balanced_directed_lightning_network_fees_3_5000_rebalancing_operations"
balances_file_name = "finalResults/fullExperimentNonStrictRebalancing/better_balanced_directed_lightning_network_fees_3_5000_step_144"
# experiment_name = "test"


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
        self.ginis = {}
        self.zetas = {}

        f = open(file_name, "r")
        for line in f:
            fields = line[:-1].split("\t")
            if len(fields) == 6:
                self.__add_channel_with_fees(fields)
        print(len(self.G))
        self.__compute_rebalance_Graph()
        self.__compute_metrics()
        print("imported a graph with {} nodes and {} edges ".format(
            len(self.G), len(self.G.edges())/2))

        # apsp = nx.all_pairs_bellman_ford_path(self.G, weight="base")
        if recompute_all_pair_shortest_paths:
            apsp = nx.all_pairs_dijkstra_path(self.G, weight="base")
            sp = open("all_pair_shortest_paths", "w")
            splengths = {}
            for i, paths in enumerate(apsp):
                print(i, paths[0], len(paths[1]))
                for k, v in paths[1].items():
                    sp.write("\t".join(v)+"\n")
                    l = len(v)
                    if l in splengths:
                        splengths[l] += 1
                    else:
                        splengths[l] = 1
                sp.flush()
                print(i, splengths)
            sp.close()
        print(
            "computed_shortestpaths on the base fee graph for evaluating routing requests")

    def __update_channels(self, circle, amt):
        ptr = 1
        fees = 0
        while ptr < len(circle):
            src = circle[ptr-1]
            dest = circle[ptr]
            ptr += 1
            self.G[src][dest]["balance"] -= amt
            self.G[dest][src]["balance"] += amt
            self.zetas[src][dest] = self.__compute_zeta(src, dest)
            self.ginis[src] = self.gini(list(self.zetas[src].values()))
            # don't forget to update the other direction!
            src, dest = dest, src
            self.zetas[src][dest] = self.__compute_zeta(src, dest)
            self.ginis[src] = self.gini(list(self.zetas[src].values()))

    def simulate_precomputed_rebalance_operations(self, file_name):
        f = open(file_name, "r")
        imbalance_scores = []
        for line in f:
            amt, circle = line[:-1].split("\t")
            amt = float(amt)
            circle = circle.split(" ")
            self.__update_channels(circle, amt)
            imbalance_scores.append(self.imbalance())
            if len(imbalance_scores) % 100 == 0:
                print(len(imbalance_scores), imbalance_scores[-1])
        return imbalance_scores

    def gini(self, x):
        # FIXME: replace with a more efficient implementation
        mean_absolute_differences = np.abs(np.subtract.outer(x, x)).mean()
        # print(x)
        relative_absolute_mean = mean_absolute_differences/np.mean(x)
        # print(relative_absolute_mean)
        return 0.5 * relative_absolute_mean

    def __compute_zeta(self, u, v):
        balance = self.G[u][v]["balance"]
        capacity = self.G[u][v]["capacity"]
        zeta = balance / capacity
        return zeta

    def __compute_metrics(self):
        for u in self.G:
            zetas = {}
            for v in self.G[u]:
                zetas[v] = self.__compute_zeta(u, v)
            self.zetas[u] = zetas
            g = self.gini(list(zetas.values()))
            self.ginis[u] = g

    def imbalance(self):
        return np.mean(list(self.ginis.values()))

    def get_gini_distribution(self):
        self.__compute_metrics()
        return list(self.ginis.values())

    def load_balance(self, file_name):
        f = open(file_name, "r")
        for l in f:
            vals = l[:-1].split("\t")
            src = vals[0]
            dest = vals[1]
            cap = int(vals[2])
            balance = int(vals[3])
            if self.G[src][dest]["capacity"] == cap:
                self.G[src][dest]["balance"] = balance
            else:
                print("!!!")
                print(line)
                exit()

    def __compute_rebalance_amount(self, circle):
        ptr = 1
        amt = sys.maxsize
        # FIXME: better way to explain -1 as channel propagation scheme
        while ptr < len(circle):
            src = circle[ptr-1]
            dest = circle[ptr]
            ptr += 1
            tmp = self.G[src][dest]["balance"]
            if tmp < amt:
                amt = tmp
        return amt

    def evaluate_routing_paths(self):
        lengths = []
        amts = []
        f = open("all_pair_shortest_paths", "r")
        for l in f:
            path = l[:-1].split("\t")
            l = len(path)
            lengths.append(l)
            if len(lengths) % 100000 == 0:
                print(len(lengths))
            amt = self.__compute_rebalance_amount(path)
            amts.append(amt)
        return lengths, amts

    def plot_payable_amt_histogram(self, lengths, balanced, unbalanced):
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(lengths, bins=[0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10], cumulative=True, normed=True, histtype="step", linestyle=("solid"),  linewidth=3)
        plt.xlabel("Distance (d) edges")
        plt.ylabel("Cumulative distribution ($P(x\geq d)$)")
        plt.title(
            "Distance distribution of shortest paths on the base fee graph")
        plt.yscale("log")
        plt.xlim(0, 9.5)
        plt.grid()
        plt.savefig("fig/cummulative_distance_distribution_log_scale.png")
        plt.close()

        plt.figure(figsize=(6.5, 4.5))
        plt.hist(balanced, bins=list(range(0, 17000000, 1000)), cumulative=True,
                 normed=True, histtype="step", linestyle=("solid"),  linewidth=3, label="Balanced Network (G = 0.188)")
        plt.hist(unbalanced, bins=list(range(0, 17000000, 10000)), cumulative=True,
                 normed=True, histtype="step", linestyle=("solid"),  linewidth=3, label="Imbalanced Network G = 0.497")

        plt.xlabel("Maximum possible payable amount (a) [sat]")
        plt.ylabel("Cumulative distribution ($P(x\geq a)$)")
        plt.title("Distribution of payable amounts on all pair cheapest paths")
        plt.xlim(0, 300000)
        # plt.yscale("log")
        # plt.xscale("log")
        plt.grid()
        plt.legend(loc="lower right")
        plt.savefig(
            "fig/maximum_payable_amount_all_pair_chepest_paths_balanced_network.png")


def save_array(arr, name):
    f = open(name, "w")
    for val in arr:
        f.write("{}\n".format(val))
    f.flush()
    f.close()


def open_array(name):
    arr = []
    f = open(name, "r")
    for line in f:
        arr.append(int(line[:-1]))
    return arr


n = Network(dataset)
recalc = False
if recalc:
    lengths, unbalanced = n.evaluate_routing_paths()
    save_array(unbalanced, "unbalanced_payment_dist")
    save_array(lengths, "length_dist")
    n.load_balance(balances_file_name)
    lenghts, balanced = n.evaluate_routing_paths()
    save_array(balanced, "balanced_payment_dist")

recalc_payment_expected_value = False
if recalc_payment_expected_value:
    lengths = open_array("length_dist")
    unbalanced = open_array("unbalanced_payment_dist")
    balanced = open_array("balanced_payment_dist")
    n.load_balance(balances_file_name)
    n.plot_payable_amt_histogram(lengths, balanced, unbalanced)

unbalanced_ginis = n.get_gini_distribution()
print(n.imbalance())
# plt.hist(unbalanced_ginis)
# plt.show()

n.load_balance(balances_file_name)
balanced_ginis = n.get_gini_distribution()

#plt.hist(balanced_ginis, label="Balanced Network (G = 0.188)")
# plt.show()
bins = [0.01 * x for x in range(100)]
plt.title("Comparing Imbalance scores of nodes in Networks")
ba_res, _, _ = plt.hist(balanced_ginis, bins=bins, cumulative=True,
                        normed=True, histtype="step", linestyle=("solid"),  linewidth=3, label="Balanced Network (G = 0.188)")
unba_res, _, _ = plt.hist(unbalanced_ginis, bins=bins, cumulative=True,
                          normed=True, histtype="step", linestyle=("solid"),  linewidth=3, label="Imbalanced Network (G = 0.497)")

m = 0
idx = 0
for i in range(len(ba_res)):
    t = abs(ba_res[i] - unba_res[i])
    if t > m:
        m = t
        idx = i
print(m, idx)

plt.plot([idx*0.01, idx*0.01], [unba_res[idx], ba_res[idx]],
         linewidth=3, label="Kolmogorov Smirnoff Distance = {:4.2f}".format(m))

plt.grid()
plt.xlabel("Imbalance (Ginicoefficients $G_v$) of nodes")
plt.ylabel("Cumulative distribution ($P(x\geq G_v)$)")
plt.xlim(0, 0.95)
plt.legend(loc="lower right")
plt.savefig(
    "fig/comparison distribution of Ginicoefficients.png")

plt.show()

exit()
w = open(experiment_name + "_imbascores_per_rebalance", "w")
for imbalance in imba:
    w.write(imbalance + "\n")
w.flush()
w.close()

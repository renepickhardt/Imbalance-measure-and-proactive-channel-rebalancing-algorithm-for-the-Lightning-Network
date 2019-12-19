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
experiment_name = "finalResults/better_balanced_directed_lightning_network_fees_4_5000_rebalancing_operations"

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

    def get_nu_dist(self):
        return list(self.nu.values())

    def simulate_precomputed_rebalance_operations(self, file_name):
        f = open(file_name, "r")
        w = open(experiment_name + "_imbascores_per_rebalance", "w")
        stats = open(experiment_name + "_stats", "w")
        hist = {}
        for cnt, line in enumerate(f):
            amt, circle = line[:-1].split("\t")
            amt = float(amt)
            circle = circle.split(" ")
            self.__update_channels(circle, amt)
            imbalance = self.imbalance()
            key = "{0:.2f}".format(imbalance)
            if key not in hist:
                _, amts = self.evaluate_routing_paths()
                hist[key] = amts
                z = 0
                for amt in amts:
                    if amt == 0:
                        z += 1
                print(key, np.median(amts), z, z/len(amts))
                stats.write("{}\t{}\t{}\t{}\n".format(
                    key, np.median(amts), z, z/len(amts)))
                stats.flush()
            w.write("{}\n".format(imbalance))

            if cnt % 1000 == 0:
                w.flush()
                print(cnt, imbalance)
        w.close()
        stats.close()
        return hist

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
            if len(lengths) % 250000 == 0:
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
        plt.close()


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


def show_lightning_has_some_outlier_high_fees():
    f = open("directed_lightning_network", "r")
    for l in f:
        base = int(l[:-1].split("\t")[5])
        if base > 100000:
            print(base)


n = Network(dataset)
# n.simulate_precomputed_rebalance_operations(experiment_name)
exit()


def parse_stats_results(filename):
    f = open(filename, "r")
    imba_scores = []
    median_payments = []
    failure_rates = []
    success_rates = []
    for line in f:
        vals = line[:-1].split()
        imba_scores.append(float(vals[0]))
        median_payments.append(int(float(vals[1])))
        failure_rates.append(float(vals[3]))
        success_rates.append(1 - failure_rates[-1])
    return (imba_scores, median_payments, failure_rates, success_rates)


# define keys for experiments
mpp = "mpp-rebalancing"
foaf = "foaf-rebalancing"
cycle4 = "cycles of lenght 4"
cycle5 = "cycles of length 5"
results = {mpp: {}, foaf: {}, cycle4: {}, cycle5: {}}
# define keys for statistics
imbalance_measure = "imbalance_measure"
median_payments = "median_payments"
failure_rates = "failure_rates"
success_rates = "success_rates"

results[mpp][imbalance_measure], results[mpp][median_payments], results[mpp][failure_rates], results[mpp][success_rates] = parse_stats_results(
    "finalResults/statistics/better_balanced_directed_lightning_network_fees_3_5000_mpprebalancing_operations_stats")
results[foaf][imbalance_measure], results[foaf][median_payments], results[foaf][failure_rates], results[foaf][success_rates] = parse_stats_results(
    "finalResults/statistics/better_balanced_directed_lightning_network_fees_3_5000_rebalancing_operations_stats")
results[cycle4][imbalance_measure], results[cycle4][median_payments], results[cycle4][failure_rates], results[cycle4][success_rates] = parse_stats_results(
    "finalResults/statistics/better_balanced_directed_lightning_network_fees_3_5000_strictrebalancing_operations_stats")
results[cycle5][imbalance_measure], results[cycle5][median_payments], results[cycle5][failure_rates], results[cycle5][success_rates] = parse_stats_results(
    "finalResults/statistics/better_balanced_directed_lightning_network_fees_4_5000_rebalancing_operations_stats")
print("done")
keys = [mpp, foaf, cycle4,  cycle5]
for key in keys:
    plt.plot(results[key][imbalance_measure],
             results[key][median_payments], label=key, linewidth=3)
plt.title("Comparing Network imbalance with possible payment size")
plt.xlabel("Network imbalance (G)")
plt.ylabel("Median possible payment size [satoshi]")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("fig/imba_vs_median_payment_size.png")
plt.close()

for key in keys:
    plt.plot(results[key][imbalance_measure],
             results[key][failure_rates], label=key, linewidth=3)
plt.title("Comparing Network imbalance with failure rate of random payments")
plt.xlabel("Network imbalance (G)")
plt.ylabel("Failure rate of random payment")
plt.grid()
plt.legend(loc="upper left")
plt.savefig("fig/imba_vs_failure_rates.png")
plt.close()

for key in keys:
    plt.plot(results[key][imbalance_measure],
             results[key][success_rates], label=key, linewidth=3)
plt.title("Comparing Network imbalance with success rate of random payments")
plt.xlabel("Network imbalance (G)")
plt.ylabel("Success rate of random payment")
plt.grid()
plt.legend(loc="lower left")
plt.savefig("fig/imba_vs_success_rates.png")
plt.close()


def parse_imbalance_scores(filename):
    f = open(filename, "r")
    return [float(l[:-1]) for l in f]


steps = "steps"
results[mpp][steps] = parse_imbalance_scores(
    "finalResults/better_balanced_directed_lightning_network_fees_3_5000_mpprebalancing_operations_imbascores_per_rebalance")
results[cycle4][steps] = parse_imbalance_scores(
    "finalResults/better_balanced_directed_lightning_network_fees_3_5000_strictrebalancing_operations_imbascores_per_rebalance")
results[cycle5][steps] = parse_imbalance_scores(
    "finalResults/better_balanced_directed_lightning_network_fees_4_5000_rebalancing_operations_imbascores_per_rebalance")
results[foaf][steps] = parse_imbalance_scores(
    "finalResults/fullExperimentNonStrictRebalancing/better_balanced_directed_lightning_network_fees_3_5000_rebalancing_operations_imbascores_per_rebalance")

for key in keys:
    plt.plot(results[key][steps], label=key, linewidth=3)
plt.title("Network imbalance over time (successfull rebalancing operations)")
plt.xlabel("Number of successfull rebalancing operations (logarithmic)")
plt.ylabel("Network imbalance (G)")
plt.xscale("log")
plt.xlim(100, 10000000)
plt.grid()
plt.legend(loc="upper right")
plt.savefig("fig/imba_vs_steps.png")
plt.close()

exit()

f = open("finalResults/fullExperimentNonStrictRebalancing/better_balanced_directed_lightning_network_fees_3_5000_fees", "r")
fees = []
tp = 0
fn = 0
for l in f:
    fee = float(l[:-1].split("\t")[1])
    # if abs(fee) > 100000000:
    if abs(fee) > 10000:
        print(fee)
        fn += 1
    else:
        fees.append(fee/1000)
        tp += 1
print(max(fees), tp, fn)


plt.hist(fees, bins=20)
plt.title("Distribution of earned / spend fees of nodes while rebalancing")
plt.xlabel("earned fees $x$ [satoshis]")
plt.ylabel("Frequency $C(x)$")
plt.grid()
plt.savefig("fig/distribution_of_fees.png")
plt.close()
# exit()

n = Network(dataset)
nus = n.get_nu_dist()
plt.hist(nus, bins=20)
plt.grid()
plt.xlabel("Node's balance coefficient $ \\nu = \\frac{\\tau}{\\kappa} $")
plt.ylabel("Frequency $ C(\\nu) $")
plt.title("Distribution of relative funds across the network")
plt.savefig(
    "fig/distribution_of_nus.png")
# plt.show()
plt.close()

# exit()
recalc = False
if recalc:
    lengths, unbalanced = n.evaluate_routing_paths()
    save_array(unbalanced, "unbalanced_payment_dist")
    save_array(lengths, "length_dist")
    n.load_balance(balances_file_name)
    lenghts, balanced = n.evaluate_routing_paths()
    save_array(balanced, "balanced_payment_dist")

recalc_payment_expected_value = True
if recalc_payment_expected_value:
    lengths = open_array("length_dist")
    unbalanced = open_array("unbalanced_payment_dist")
    balanced = open_array("balanced_payment_dist")
    n.load_balance(balances_file_name)
    n.plot_payable_amt_histogram(lengths, balanced, unbalanced)

n = Network(dataset)
unbalanced_ginis = n.get_gini_distribution()
print(n.imbalance())
plt.hist(unbalanced_ginis, bins=20)
plt.title("Initial Distribution of nodes Ginicoefficients (imbalanced Network)")
plt.xlabel("Imbalance (Ginicoefficients $G_v$) of nodes")
plt.ylabel("Frequency $C(G_v)$")
plt.grid()
plt.savefig("fig/initial_ginis_before_rebalancing.png")
plt.close()

n.load_balance(balances_file_name)
balanced_ginis = n.get_gini_distribution()

plt.hist(balanced_ginis, bins=20, label="Balanced Network (G = 0.188)")
plt.title("Final Distribution of nodes Ginicoefficients (balanced Network)")
plt.xlabel("Imbalance (Ginicoefficients $G_v$) of nodes")
plt.ylabel("Frequency $C(G_v)$")
plt.grid()
plt.savefig("fig/Final_ginis_after_rebalancing.png")
plt.close()

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

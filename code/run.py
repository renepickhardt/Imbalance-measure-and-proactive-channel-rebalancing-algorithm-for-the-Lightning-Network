import networkx as nx
import numpy as np

balance = "balance"
capacity = "capacity"

class Network:
    G = None

    def __init__(self):
        self.G = nx.DiGraph()

    def __initialize_balances(self):
        to_add = []
        for u in self.G:
            for v in self.G[u]:
                if (v,u) not in self.G:
                    c = self.G[u][v][capacity]
                    to_add.append((v,u,c))
        for v,u,c in to_add:
            self.G.add_edge(v,u,capacity = c, balance = 0)
            
    def load_network(self):
        f = open("network","r")
        for line in f:
            f,t,c = line[:-1].split("\t")
            self.G.add_edge(f,t,capacity=int(c),balance=int(c))
        self.__initialize_balances()

    def compute_tau(self,node):
        tc = 0
        tb = 0
        try:
            for n in self.G[node]:
                tc += self.G[node][n][capacity]
                tb += self.G[node][n][balance]
        except:
            print(node, self.G)
        return float(tb)/float(tc)

    def __beta(self, u, v):
        return float(self.G[u][v][balance])/float(self.G[u][v][capacity])

    def beta(self,u,v):
        return self.__beta(u,v)
    
    def compute_betas(self,node):
        return [self.__beta(node,n) for n in self.G[node]]
            
    def gini(self, x):
        #FIXME: replace with a more efficient implementation
        mean_absolute_differences = np.abs(np.subtract.outer(x,x)).mean()
        relative_absolute_mean = mean_absolute_differences/np.mean(x)
        return 0.5 * relative_absolute_mean

    def total_health(self):
        return np.mean([self.gini(self.compute_betas(n)) for n in self.G])


    def __get_outbound_channels(self, node):
        tau = self.compute_tau(node)
        return [n for n in self.G[node] if self.__beta(node,n) > tau]

    def __get_inbound_channels(self, node):
        tau = self.compute_tau(node)
        return [n for n in self.G[node] if self.__beta(node,n) < tau]
        
    
    
    
    def __find_foaf_circles(self,u,v):
        tmpG = nx.DiGraph()
        out_nodes = self.__get_outbound_channels(v)
        for x in out_nodes:
            tmpG.add_edge(v,x)
        in_nodes = self.__get_inbound_channels(u)
        #print(in_nodes, u, v, out_nodes)
        for x in in_nodes:
            tmpG.add_edge(x,u)
            foaf_in = self.__get_inbound_channels(x)
            for y in foaf_in:
                tmpG.add_edge(y,x)
        if u in tmpG and v in tmpG:
            path = []
            try:
                path = nx.dijkstra_path(tmpG, v, u)
            except: #probably no path / circle exist
                pass
            res = [u]
            res.extend(path)
            return res
        return []

    def __rebalance_circle(self, circle, amt):
        for i in range(len(circle)-1):
            f = circle[i]
            t = circle[i+1]
            self.G[f][t][balance]-=amt
            self.G[t][f][balance]+=amt


    def __compute_rebalance_amt(self, circle):
        """
        cannot be done in the real lightning network like this

        since we do a simulation it is fine to compute this sloppy
        """
        amt = 100000
        for i in range(len(circle)-1):
            f = circle[i]
            t = circle[i+1]
            tau = np.mean(self.compute_betas(f))
            #tau = self.compute_tau(f)
            beta = self.__beta(f, t)
            #FIXME: Maybe balance?
            tmp = int(np.abs(tau - beta) * self.G[f][t][capacity])
            if tmp < amt:
                amt = tmp
            
        return amt
#        
    
    def make_node_healthier(self, node):
        tau = self.compute_tau(node)
        candidate = None
        max_diff = 0
        for n in self.G[node]:
            beta = self.__beta(node, n)
            diff = beta - tau
            if diff > max_diff:
                max_diff = diff
                candidate = n
        if candidate is None:
            print("Node {} has no candidates for rebalancing right now".format(node))
            print(self.G[node])
            return
        #FIXME: check if we made sure that the inbound channel has a smaller beta_value than gamma / tau
        circle = self.__find_foaf_circles(node,candidate)
        amt = self.__compute_rebalance_amt(circle)
        if len(circle) < 3:
            return
        print(amt, circle)
        self.__rebalance_circle(circle,7)
        #print(node, circle)


n = Network()
n.load_network()
#n.G.add_edge("B","A",capacity=100,balance=0)

import random


raw_data = open("raw", "w")


nodes = list(n.G)

health = n.total_health()
health_ts = [health]
for step in range(250):
    node = random.choice(nodes)
    n.make_node_healthier(node)
    health = n.total_health()
    raw_data.write("{}\t{}\n".format(node,health))
    health_ts.append(health)
    #print(step)

raw_data.write("\n")
for node in n.G:
    #raw_data.write(node)
    for adj in n.G[node]:
        raw_data.write("{}\t{}\t{}\t{}\t{}\n".format(node, adj,n.G[node][adj][capacity], n.G[node][adj][balance],n.beta(node,adj)))

raw_data.write("\n")
for node in nodes:
    raw_data.write("{}\t{}\n".format(node,n.gini(n.compute_betas(node))))
raw_data.flush()
raw_data.close()

import matplotlib.pyplot as plt
plt.plot(health_ts)
plt.title("Network health measured as evenly distributed liquidity")
plt.ylabel("average Ginicoefficient of local chanal balance coefficients")
plt.xlabel("simulation steps")
plt.grid()
plt.show()

for node in n.G:
    betas = n.compute_betas(node)
    print(node, n.G[node], n.compute_tau(node), betas, n.gini(betas))
    n.make_node_healthier(node)
print(n.total_health())

for node in n.G:
    betas = n.compute_betas(node)
    print(node, n.G[node], n.compute_tau(node), betas, n.gini(betas))

c = random.choice(list(n.G))
print(c)
c = random.choice(list(n.G))
print(c)
c = random.choice(list(n.G))
print(c)

print(n.gini([100,100,100,200]))
    

#n.compute_balances()
#for node in n.G:
#    print(n.G[node])

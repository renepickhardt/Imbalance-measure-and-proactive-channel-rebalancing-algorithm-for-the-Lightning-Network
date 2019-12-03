"""
Author: Rene Pickhardt
Date: Dec. 3rd 2019

bThis script goes through a snapshot of the lightning_network (from
c-lightning) and randomly assigns a direction to the edges.

On the reslting Directed graph it selects the largest strongly
connected component and adds also the reverse channel fee policies to
the data set.

"""

import random
import json
import networkx as nx
f = open("../channels.json", "r")

channels = json.load(f)
nodes = set()
edges = set()

# count if each channel occures twice!
sid_cnt = {}
for channel in channels["channels"]:
    sid = channel["short_channel_id"]
    if sid in sid_cnt:
        sid_cnt[sid] += 1
    else:
        sid_cnt[sid] = 1

for channel in channels["channels"]:
    s = channel["source"]
    d = channel["destination"]
    sid = channel["short_channel_id"]
    # only include channels that exist in both directions
    if sid_cnt[sid] != 2:
        continue
    nodes.add(s)
    nodes.add(d)
    if s > d:
        s, d = d, s
    edges.add((s, d))

print(len(nodes), len(edges))


def direction(src, dest):
    r = random.randint(0, 1)
    if r > 0:
        return (src, dest)
    return (dest, src)


used_edges = set([direction(*edge) for edge in edges])
used_edges_with_metadata = set()

for channel in channels["channels"]:
    s = channel["source"]
    d = channel["destination"]
    a = channel["satoshis"]
    e = (s, d)
    base = channel["base_fee_millisatoshi"]
    rate = channel["fee_per_millionth"]
    r = (d, s)
    if e in used_edges:
        used_edges_with_metadata.add((s, d, a, a, base, rate))


def find_channel(s, d):
    for channel in channels["channels"]:
        if s == channel["source"] and d == channel["destination"]:
            a = channel["satoshis"]
            base = channel["base_fee_millisatoshi"]
            rate = channel["fee_per_millionth"]
            return s, d, a, 0, base, rate
    return False


print(len(used_edges), len(used_edges_with_metadata))

G = nx.DiGraph()
for s, d, c, b, base, rate in used_edges_with_metadata:
    G.add_edge(s, d, capacity=c, balance=b, base=base, rate=rate)

res = list(nx.strongly_connected_components(G))
# print(len(res))
scc = []
for c in res:
    if len(c) > 10:
        scc = c
        print(len(c))
H = G.subgraph(scc)

print(len(H))
w = open("directed_lightning_network", "w")
for s in H:
    for d in H[s]:
        w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
            s, d, G[s][d]["capacity"], G[s][d]["balance"],
            G[s][d]["base"], G[s][d]["rate"], ))
        res = find_channel(d, s)
        if type(res) == bool:
            print("this can only be seen if there was a semantic error
                  in the programm")
            continue
        src, dest, cap, bal, base, rate = find_channel(d, s)
        w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(src, dest, cap, bal,
                                                  base, rate))

        # for e in reversed_edges_with_metadata:
        #    if e[0] == d and e[1] == s:
        #        w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(*e))
        #        break
        w.flush()
w.close()

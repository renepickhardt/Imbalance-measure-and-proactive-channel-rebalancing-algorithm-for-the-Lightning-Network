import json
import networkx as nx
f = open("channels.json", "r")

nodes = set()
used_channels = []
channels = json.load(f)

uniset = set()


def add_channel(a, b, c):
    tmp = a+b
    if a > b:
        tmp = b + a
    if tmp in uniset:
        return
    uniset.add(tmp)
    used_channels.append((a, b, c))


for num, channel in enumerate(channels["channels"]):
    s = channel["source"]
    d = channel["destination"]
    if len(nodes) > 2000:
        if s in nodes and d in nodes:
            add_channel(s[0:7], d[0:7], channel["satoshis"])
            #used_channels.append((s[0:7], d[0:7], channel["satoshis"]))
    else:
        add_channel(s[0:7], d[0:7], channel["satoshis"])
        #used_channels.append((s[0:7], d[0:7], channel["satoshis"]))
        nodes.add(s)
        nodes.add(d)
#    if num > 17000:
#        break


#print(channel, len(nodes), len(used_channels))


snodes = set([n[0:7] for n in nodes])
# print(len(snodes))

G = nx.DiGraph()
for s, d, a in used_channels:
    G.add_edge(s, d, balance=a)

res = list(nx.strongly_connected_components(G))
# print(len(res))
scc = []
for c in res:
    if len(c) > 10:
        scc = c
        print(len(c))
# print(go)
print("03efccf" in scc)


H = G.subgraph(scc)

w = open("lightning_network", "w")
for s in H:
    for d in H[s]:
        w.write("{}\t{}\t{}\n".format(s, d, G[s][d]["balance"]))

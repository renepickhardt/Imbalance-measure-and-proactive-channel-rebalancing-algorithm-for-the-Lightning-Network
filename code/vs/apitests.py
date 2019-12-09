"""
this is just for playing arround with the networkx API in order to make sure I understood it correctly
"""

import networkx as nx

G = nx.DiGraph()

G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 0)
G.add_edge(1, 4)
G.add_edge(4, 3)
G.add_edge(1, 5)
G.add_edge(5, 6)
G.add_edge(6, 3)

for path in nx.all_simple_paths(G, 1, 0, 3):
    print(path)

for c in nx.simple_cycles(G):
    print(c)

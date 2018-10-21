import networkx as nx
import matplotlib.pyplot as plt
G = nx.read_adjlist('../dataset/karate.adjlist')

nx.draw(G)
plt.show()
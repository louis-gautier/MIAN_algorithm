import numpy as np
import sys

class Greedy:
    def __init__(self, G, kmax, estimate_PIS, results_file):
        self.G = G
        self.kmax = kmax # Maximum number of seed nodes
        self.estimate_PIS = estimate_PIS # Function to estimate the value of a set of seed nodes
        self.results_file = results_file

    def run(self):
        result = set()
        V = set(self.G.nodes)
        n_i = self.kmax
        for i in range(self.kmax):
            p_i = (i + 1) / n_i
            # print("k=", i)
            node_values = {}
            n_j = len(V.difference(result))
            for j, w in enumerate(V.difference(result)):
                p_j = (j + 1) / n_j
                sys.stdout.write('\r')
                sys.stdout.write("k: [%-20s] %d%%" % ('='*int(20*p_i), 100*p_i))
                sys.stdout.write(" | w: [%-20s] %d%%" % ('='*int(20*p_j), 100*p_j))
                sys.stdout.flush()
                # print("j", j)
                node_values[w] = self.estimate_PIS(result | {w})
            u = max([v for v in node_values.keys()], key=lambda x: node_values[x])
            result.add(u)
            with open(self.results_file, 'a') as results_file:
                results_file.write(str(u)+'\n')
        return result
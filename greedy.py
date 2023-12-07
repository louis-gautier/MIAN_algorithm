import numpy as np

class Greedy:
    def __init__(self, G, kmax, estimate_PIS, results_file):
        self.G = G
        self.kmax = kmax # Maximum number of seed nodes
        self.estimate_PIS = estimate_PIS # Function to estimate the value of a set of seed nodes
        self.results_file = results_file

    def run(self):
        result = set()
        V = set(self.G.nodes)
        for i in range(self.kmax):
            print("k=", i)
            node_values = {}
            for j, w in enumerate(V.difference(result)):
                print("j", j)
                node_values[w] = self.estimate_PIS(result | {w})
            u = max([v for v in node_values.keys()], key=lambda x: node_values[x])
            result.add(u)
            with open(self.results_file, 'a') as results_file:
                results_file.write(str(u)+'\n')
        return result
import numpy as np

class Greedy:
    def __init__(self, kmax, estimate_PIS, results_file):
        self.kmax = kmax # Maximum number of seed nodes
        self.estimate_PIS = estimate_PIS # Function to estimate the value of a set of seed nodes
        self.results_file = results_file

    def run(self):
        result = set()
        V = set(self.G.nodes)
        for i in range(self.kmax):
            node_values = [self.estimate_PIS(result | {w}) for w in V.difference(result)]
            u = np.argmax(node_values)
            result.add(u)
            with open(self.results_file, 'a') as results_file:
                results_file.write(str(u)+'\n')
        return result
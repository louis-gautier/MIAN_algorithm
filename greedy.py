import numpy as np

class Greedy:
    def __init__(self, k, f, G):
        self.k = k # Maximum number of seed nodes
        self.f = f # Function to estimate the value of a set of seed nodes
        self.G = G # Directed Networkx Graph

    def run(self):
        result = set()
        V = set(self.G.nodes)
        for i in range(self.k):
            u = np.argmax([self.f(result | {w}) - self.f(result) for w in V.difference(result)])
            result.add(u)
        return result
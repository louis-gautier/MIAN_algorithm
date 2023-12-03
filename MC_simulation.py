import numpy as np
import networkx as nx

def __init__(self, G, S, q):
    self.G = G
    self.q = q
    self.S = S

    self.state = {v: 0 for v in G.nodes}
    for v in G.nodes:
        if np.random.rand()<=q:
            self.state[v] = 1 #state(v) = positive with prob. q
        else:
            self.state[v] = -1 #otherwise negative
    
    while self.S!=set():
        Snew = set()
        # Make the Set Sucessors, Union of u in Si: N^{out}(u)
        Sucessors = set()
        for u in self.S:
            Sucessors = Sucessors.union(G.sucessors(u))
        # For each sucessor v, determine its activation status and its state
        for v in Sucessors:
            if self.state[v]==0: #and state(v) = neutral
                rho = list(self.S.union(G.predecessors(v))) #rho is union of Si and N^{in}(v)
                np.random.shuffle(rho) #order rho uniformly at random!!!
                for u in rho:
                    # Based on the activation probability p(u,v), v is added to Snew, and state is determined
                    if np.random.rand()<=G.get_edge_data(u,v)["weight"]:
                        Snew = Snew.union(v) #v is activated by u with prob. p(u,v)
                        if self.state[u]==1: #if state(u) = positive, v positive with prob. q
                            if np.random.rand()<= q:
                                self.state[v] = 1 #state(v) = positive with prob. q
                            else:
                                self.state[v] = -1 #otherwise state(v) = negative
                        else: #if state(u) = negative, v always negative
                            self.state[v] = -1 #state(v) = negative
        self.S = Snew
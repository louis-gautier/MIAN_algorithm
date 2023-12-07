import numpy as np
import networkx as nx
from graphs import get_graph

class ICN:
    def __init__(self, G, S, q):
        self.G = G
        self.S = S
        self.q = q

        self.state = {u: 0 for u in self.G.nodes}
        for v in self.S:
            if np.random.rand()<=q:
                self.state[v] = 1 #state(v) = positive with prob. q
            else:
                self.state[v] = -1 #otherwise negative
        while len(self.S)>0:
            Snew = set()
            # Make the Set Sucessors, Union of u in Si: N^{out}(u)
            Sucessors = set()
            for u in self.S:
                Sucessors = Sucessors.union(self.G.successors(u))
            # For each sucessor v, determine its activation status and its state
            for v in Sucessors:
                if self.state[v]==0: #and state(v) = neutral
                    rho = list(self.S.intersection(self.G.predecessors(v))) #rho is union of Si and N^{in}(v)
                    np.random.shuffle(rho) #order rho uniformly at random!!!
                    for u in rho:
                        # Based on the activation probability p(u,v), v is added to Snew, and state is determined
                        if np.random.rand() <= self.G.get_edge_data(u,v)["weight"]:
                            Snew.add(v) #v is activated by u with prob. p(u,v)
                            if self.state[u]==1: #if state(u) = positive, v positive with prob. q
                                if np.random.rand() <= self.q:
                                    self.state[v] = 1 #state(v) = positive with prob. q
                                else:
                                    self.state[v] = -1 #otherwise state(v) = negative
                            else:
                                assert self.state[u]==-1 #if state(u) = negative, v always negative
                                self.state[v] = -1 #state(v) = negative
            self.S = Snew
    
    def print_state(self):
        print(f"positive nodes: {np.sum(self.state.values() == 1)}")
        print(f"neutral nodes: {np.sum(self.state.values() == 0)}")
        print(f"negative nodes: {np.sum(self.state.values() == -1)}")

    def positive_influence_spread(self):
        return np.sum(self.state.values()==1)

if __name__ == '__main__':
    G = get_graph("epinions_subgraph")
    S = set(np.random.choice(G.nodes, size=3, replace=False))
    # for u in S:
    #     for v in G.successors(u):
    #         print(G.get_edge_data(u,v))
    #         raise NotImplementedError
    q=0.7
    icn = ICN(G, S, q)
    icn.print_state()
    print(icn.state)
import heapq
import networkx as nx
import numpy as np


class MIAN:
    def __init__(self, G, q, k, theta):
        self.G = G
        self.q = q
        self.k = k
        self.theta = theta

        # Initialization from the pseudo algorithm
        self.S = set()
        self.actual = {v: 0 for v in G.nodes}
        self.MIIAs = {}
        self.MIOAs = {}
        self.incinf_triangular = {{} for v in G.nodes}
        self.incinf = {}
        for v in G.nodes:
            MIIA =  self.MIIA(v)
            MIOA = self.MIOA(v)
            self.MIIAs[v] = MIIA
            self.MIOAs[v] = MIOA
        # Compute paps
        for v in G.nodes:
            for u in self.MIOAs[v]:
                dist, path = self.shortest_path(u, v, restriction=self.MIIAs[u])
                pap = dist*self.q if len(path) > 0 else 0
                self.incinf_triangular[v][u] = pap
            self.incinf[v] = np.sum(self.incinf_triangular[v].values())

    def run(self):
        for i in range(self.k):
            u = max(self.incinf, key=self.incinf.get)
            self.S.add(u)
            for v in self.MIOAs[u]:
                self.actual[v] += self.incinf_triangular[u][v]
                for w in self.MIIAs[v]:
                    # TODO: Estimate papv
                    papv = None
                    Delta = papv - self.actual[v]
                    self.incinf[w] += Delta - self.incinf_triangular[w][v]
                    self.incinf[w][v] = Delta
        return self.S
    
    def MIIA(self, v):
        arb = set()
        for u in self.G.nodes:
            ppp, new_nodes = self.shortest_path(u,v)
            if ppp >= self.theta:
                arb.update(new_nodes)
        return arb
    
    def MIOA(self, v):
        arb = set()
        for u in self.G.nodes:
            ppp, new_nodes = self.shortest_path(v,u)
            if ppp >= self.theta:
                arb.update(new_nodes)
        return arb
    
    
    def shortest_path(self, start, end, restriction=None):
        shortest_distance = {}
        predecessor = {}
        heap = []
        heapq.heappush(heap, (0, start, None)) # distance, node, previous
        
        while heap:
            distance, node, previous = heapq.heappop(heap)
            if node in shortest_distance:
                continue
            shortest_distance[node] = distance
            predecessor[node] = previous
            if node == end:
                path = []
                while node:
                    path.append(node)
                    node = predecessor[node]
                return distance, path[::-1]
            else:
                for edge in self.G.edges(node):
                    if restriction is not None and edge[1] not in restriction:
                        continue
                    dist = self.q * self.G.get_edge_data(*edge)["weight"]
                    heapq.heappush(heap, (distance + dist, edge[1], node))
        else:
            return np.inf, []

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
        self.incinf_matrix = {{} for v in G.nodes}
        self.incinf_vector = {}
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
                self.incinf_matrix[v][u] = pap
            self.incinf_vector[v] = np.sum(self.incinf_matrix[v].values())

    def run(self):
        for i in range(self.k):
            u = max(self.incinf_vector, key=self.incinf_vector.get)
            self.S.add(u)
            for v in self.MIOAs[u]:
                self.actual[v] += self.incinf_matrix[u][v]
                for w in self.MIIAs[v]:
                    papv = self.PAP(v, w, self.MIIAs[v])
                    Delta = papv - self.actual[v]
                    self.incinf_vector[w] += Delta - self.incinf_matrix[w][v]
                    self.incinf_vector[w][v] = Delta
        return self.S
    
    # def AP(self, v, t, arb):
    #     if t == 0:
    #         return int(v in self.S)
    #     if v in self.S:
    #         return 0
    #     prob_unactivated_earlier = np.product([1 - sum([self.AP(edge[1], i)*arb.get_edge_data(*edge)["weight"]
    #                                                     for i in range(max(0,t-2))])
    #                                            for edge in arb.edges(v)])
    #     prob_unactivated_by_now = np.product([1 - sum([self.AP(edge[1], i)*arb.get_edge_data(*edge)["weight"]
    #                                                    for i in range(t-1)])
    #                                        for edge in arb.edges(v)])
    #     return prob_unactivated_earlier - prob_unactivated_by_now
    
    def PAP(self, v, w, arb):
        S_tent = self.S + {w}
        h = nx.dag_longest_path_length(arb, source=v)
        n = arb.number_of_nodes()
        AP_matrix = np.zeros((n, h))
        u_in_S_mask = np.array([v in S_tent for v in arb.nodes])
        AP_matrix[u_in_S_mask,0] = 1
        AP_matrix[u_in_S_mask,1:] = 0
        AP_matrix[~u_in_S_mask,0] = 0
        for t in range(1, h):
            for u in arb.nodes:
                if u in S_tent:
                    continue
                if t > 1:
                    prob_activated_earlier = np.product([1 - sum([AP_matrix[edge[1], i]*arb.get_edge_data((w, u))["weight"]
                                                                  for i in range(t-2)])
                                                         for edge in arb.edges(v)])
                else:
                    prob_activated_earlier = 0
                prob_unactivated_by_now = np.product([1 - sum([AP_matrix[w, i]*arb.get_edge_data((w, u))["weight"]
                                                               for i in range(t-2)])
                                                      for w in arb.predecessors(u)])
                AP_matrix[u, t] = prob_activated_earlier - prob_unactivated_by_now

        return sum([AP_matrix[v,t]*self.q**(t+1) for t in range(h)])

    def MIIA(self, v):
        arb = nx.DiGraph()
        for u in self.G.nodes:
            ppp, new_nodes = self.shortest_path(u,v)
            if ppp >= self.theta:
                nx.add_path(arb, new_nodes)
        return arb
    
    def MIOA(self, v):
        arb = nx.DiGraph()
        for u in self.G.nodes:
            ppp, new_nodes = self.shortest_path(v,u)
            if ppp >= self.theta:
                nx.add_path(arb, new_nodes)
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

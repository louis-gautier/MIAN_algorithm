import heapq
import networkx as nx
import numpy as np
import time

class MIAN:
    def __init__(self, G, q, k, theta, results_file):
        # nmap = {u: i for i, u in enumerate(G.nodes)} # new mapping of nodes to integers
        # # print(G.nodes)
        # self.G = nx.relabel_nodes(G, nmap) # Make sure labels are standardized for array indexing
        # nx.convert_node_labels_to_integers
        print("Initializing MIAN")
        self.G = G
        self.q = q
        self.k = k
        self.theta = theta
        self.results_file = results_file

        # Initialization from the pseudo algorithm
        self.S = set()
        self.actual = {v: 0 for v in G.nodes}
        self.MIIAs = {}
        self.MIOAs = {}
        self.incinf_matrix = {v: {} for v in G.nodes}
        self.incinf_vector = {}
        self.hs = {}
        print("Computing initial MIIA and MIOA")
        for v in G.nodes:
            MIIA = self.MIIA(v)
            MIOA = self.MIOA(v)
            self.MIIAs[int(v)], h = MIIA
            self.hs[int(v)] = h
            self.MIOAs[int(v)] = MIOA
        # Compute paps
        print("Computing initial PAP")
        for v in G.nodes:
            for u in self.MIOAs[v]:
                dist, path = self.shortest_path(u, v, restriction=self.MIIAs[u])
                pap = dist*self.q if len(path) > 0 else 0
                self.incinf_matrix[v][u] = pap
            self.incinf_vector[v] = np.sum(list(self.incinf_matrix[v].values()))

    def run(self):
        for i in range(self.k):
            print(f"Adding {i}th seed node")
            u = max(self.incinf_vector, key=lambda x: self.incinf_vector[x])
            with open(self.results_file, 'a') as results_file:
                results_file.write(str(u)+'\n')
            self.S.add(u)
            print("Updating all PAPs")
            for v in self.MIOAs[u]:
                self.actual[v] += self.incinf_matrix[u][v]
                #print(f"Computing PAP for {v}")
                for w in self.MIIAs[v]:
                    papv = self.PAP(v, w, self.MIIAs[v])
                    Delta = papv - self.actual[v]
                    self.incinf_vector[w] += Delta - self.incinf_matrix[w][v]
                    self.incinf_matrix[w][v] = Delta
        return self.S
    
    def PAP(self, v, w, arb):
        nmap = {u: i for i, u in enumerate(arb.nodes)} # new mapping of nodes in arb
        rmap = {i: u for u, i in nmap.items()} # reverse mapping of nodes
        S_tent = self.S | {w}
        # h = nx.dag_longest_path_length(arb) # Can be optimized
        h = self.hs[int(v)]
        n = arb.number_of_nodes()
        AP_matrix = np.zeros((n, h))
        u_in_S_mask = np.array([rmap[i] in S_tent for i in range(n)])
        AP_matrix[u_in_S_mask,0] = 1
        AP_matrix[u_in_S_mask,1:] = 0
        AP_matrix[~u_in_S_mask,0] = 0
        for t in range(1, h):
            for u in arb.nodes:
                if u in S_tent:
                    continue
                if t > 1:
                    prob_activated_earlier = np.product([1 - sum([AP_matrix[nmap[w], j]*self.G.get_edge_data(w, u)["weight"]
                                                                  for j in range(t-2)])
                                                         for w in arb.predecessors(u)])
                else:
                    prob_activated_earlier = 0
                prob_unactivated_by_now = np.product([1 - sum([AP_matrix[nmap[w], j]*self.G.get_edge_data(w, u)["weight"]
                                                               for j in range(t-1)])
                                                      for w in arb.predecessors(u)])
                AP_matrix[nmap[w], t] = prob_activated_earlier - prob_unactivated_by_now
        t3 = time.time()
        return sum([AP_matrix[nmap[v],t]*self.q**(t+1) for t in range(h)])

    def MIIA(self, v):
        arb = nx.DiGraph()
        arb.add_node(v)
        for u in self.G.nodes:
            ppp, new_nodes = self.shortest_path(u,v)
            if ppp >= self.theta:
                nx.add_path(arb, new_nodes)
        h = nx.dag_longest_path_length(arb)
        return arb, h
    
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
                return np.exp(-distance), path[::-1]
            else:
                for edge in self.G.edges(node):
                    if restriction is not None and edge[1] not in restriction:
                        continue
                    dist = -np.log(self.q * self.G.get_edge_data(*edge)["weight"])
                    heapq.heappush(heap, (distance + dist, edge[1], node))
        else:
            return np.inf, []

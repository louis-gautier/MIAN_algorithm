import heapq
import networkx as nx
import numpy as np
import time

class MIAN:
    def __init__(self, G, q, k, theta, results_file):
        print("Initializing MIAN")
        self.G = G
        self.q = q
        self.k = k
        self.theta = theta
        self.results_file = results_file
        with open(self.results_file, "w") as f:
            pass
        self.S = set()
        self.actual = {int(v): 0 for v in G.nodes}
        self.MIIAs = {}
        self.MIOAs = {}
        self.incinf_matrix = {int(v): {} for v in G.nodes}
        self.incinf_vector = {}
        self.hs = {}
        
        custom_graph = G.copy()
        for edge in G.edges():
            source, target = edge
            custom_graph[source][target]["weight"] = -np.log(self.G.get_edge_data(source, target)["weight"]*q)
        print("Computing all shortest paths")
        self.all_shortest_paths = dict(nx.all_pairs_dijkstra(custom_graph)) # dict mapping each node to 2 dicts: first one is distances, second one is paths
        # WARNING: stores keys as str (takes too long to reindex everything)
        print("Completed computing all shortest paths")
        for u in G.nodes:
            for v in G.nodes:
                if u not in self.all_shortest_paths.keys() or v not in self.all_shortest_paths[u][0].keys():
                    self.all_shortest_paths[u][1][v] = []
                    self.all_shortest_paths[u][0][v] = 0
                else:
                    self.all_shortest_paths[u][0][v] = np.exp(-self.all_shortest_paths[u][0][v])

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
            for u in self.MIOAs[int(v)]:
                pap = self.all_shortest_paths[v][0][u]*self.q if len(self.all_shortest_paths[v][1][u]) > 0 else 0
                self.incinf_matrix[int(v)][int(u)] = pap
            self.incinf_vector[int(v)] = np.sum(list(self.incinf_matrix[int(v)].values()))

    def run(self):
        for i in range(self.k):
            print(f"Adding {i}th seed node")
            u = max([v for v in self.incinf_vector.keys() if v not in self.S], key=lambda x: self.incinf_vector[x])
            assert (str(u) in self.G.nodes)
            with open(self.results_file, 'a') as results_file:
                results_file.write(str(u)+'\n')
            self.S.add(int(u))
            print("Updating all PAPs")
            for i, v in enumerate(self.MIOAs[int(u)]):
                self.actual[int(v)] += self.incinf_matrix[int(u)][int(v)]
                # print(f"Computing PAPs for {i}th node")
                for w in self.MIIAs[int(v)]:
                    papv = self.PAP(v, w, self.MIIAs[int(v)])
                    Delta = papv - self.actual[int(v)]
                    try:
                        self.incinf_vector[int(w)] += Delta - self.incinf_matrix[int(w)][int(v)]
                        self.incinf_matrix[int(w)][int(v)] = Delta
                    except KeyError:
                        assert int(w) == int(v)
        return self.S
    
    def PAP(self, v, w, arb):
        nmap = {u: i for i, u in enumerate(arb.nodes)} # new mapping of nodes in arb
        rmap = {i: u for u, i in nmap.items()} # reverse mapping of nodes
        S_tent = self.S | {int(w)}
        h = self.hs[int(v)]
        n = arb.number_of_nodes()
        AP_matrix = np.zeros((n, h))
        u_in_S_mask = np.array([int(rmap[i]) in S_tent for i in range(n)])
        AP_matrix[u_in_S_mask,0] = 1
        AP_matrix[u_in_S_mask,1:] = 0
        AP_matrix[~u_in_S_mask,0] = 0
        for t in range(1, h):
            for u in arb.nodes:
                if int(u) in S_tent:
                    continue
                if t > 1:
                    prob_unactivated_earlier = np.product([1 - sum([AP_matrix[nmap[w_iter], j]*self.G.get_edge_data(w_iter, u)["weight"]
                                                                  for j in range(t-1)])
                                                         for w_iter in arb.predecessors(u)])
                else:
                    prob_unactivated_earlier = 1.0
                prob_unactivated_by_now = np.product([1 - sum([AP_matrix[nmap[w], j]*self.G.get_edge_data(w, u)["weight"]
                                                               for j in range(t)])
                                                      for w in arb.predecessors(u)])
                AP_matrix[nmap[w], t] = prob_unactivated_earlier - prob_unactivated_by_now
        return sum([AP_matrix[nmap[v],t]*self.q**(t+1) for t in range(h)])

    def MIIA(self, v):
        arb = nx.DiGraph()
        for u in self.G.nodes:
            new_nodes = self.all_shortest_paths[u][1][v]
            ppp = self.all_shortest_paths[u][0][v]
            if ppp >= self.theta:
                arb.add_weighted_edges_from([(new_nodes[i], new_nodes[i+1], self.G.get_edge_data(new_nodes[i], new_nodes[i+1])["weight"]) for i in range(len(new_nodes)-1)])
        h = nx.dag_longest_path_length(arb, weight=None)
        return arb, h
    
    def MIOA(self, v):
        arb = nx.DiGraph()
        for u in self.G.nodes:
            new_nodes = self.all_shortest_paths[v][1][u]
            ppp = self.all_shortest_paths[v][0][u]
            if ppp >= self.theta:
                arb.add_weighted_edges_from([(new_nodes[i], new_nodes[i+1], self.G.get_edge_data(new_nodes[i], new_nodes[i+1])["weight"]) for i in range(len(new_nodes)-1)])
        return arb
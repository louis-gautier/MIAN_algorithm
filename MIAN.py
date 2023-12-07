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
        with open(self.results_file, "w") as f:
            pass

        # Initialization from the pseudo algorithm
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
        all_shortest_paths = dict(nx.all_pairs_shortest_path(custom_graph))
        print("Finished computing all shortest paths")
        self.pairwise_paths = {int(u): {} for u in G.nodes}
        self.pairwise_ppp = {int(u): {} for u in G.nodes}
        for u in G.nodes:
            for v in G.nodes:
                try:
                    path = all_shortest_paths[u][v]
                    self.pairwise_paths[int(u)][int(v)] = path
                    self.pairwise_ppp[int(u)][int(v)] = np.exp(-np.sum([custom_graph.get_edge_data(path[i], path[i+1])["weight"] for i in range(len(path)-1)]))
                except KeyError:
                    self.pairwise_paths[int(u)][int(v)] = []
                    self.pairwise_distances[int(u)][int(v)] = np.inf

        print("Computing initial MIIA and MIOA")
        for v in G.nodes:
            MIIA = self.MIIA(v)
            MIOA = self.MIOA(v)
            self.MIIAs[int(v)], h = MIIA
            self.hs[int(v)] = h
            self.MIOAs[int(v)] = MIOA
        
        # print(f'Skipped {len([1 for u in self.G.nodes for v in self.G.nodes if self.pairwise_ppp[int(u)][int(v)] < self.theta])} ppps')
        # Compute paps
        print("Computing initial PAP")
        for v in G.nodes:
            # all_shortest_paths = dict(nx.all_pairs_shortest_path(self.MIOAs[int(v)]))
            for u in self.MIOAs[int(v)]:
                pap = self.pairwise_ppp[int(u)][int(v)]*self.q if len(path) > 0 else 0
                self.incinf_matrix[int(v)][int(u)] = pap
            self.incinf_vector[int(v)] = np.sum(list(self.incinf_matrix[int(v)].values()))

    def run(self):
        for i in range(self.k):
            print(f"Adding {i}th seed node")
            u = max([v for v in self.incinf_vector if v not in self.S], key=lambda x: self.incinf_vector[x])
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
                    self.incinf_vector[int(w)] += Delta - self.incinf_matrix[int(w)][int(v)]
                    self.incinf_matrix[int(w)][int(v)] = Delta
        return self.S
    
    def PAP(self, v, w, arb):
        nmap = {u: i for i, u in enumerate(arb.nodes)} # new mapping of nodes in arb
        rmap = {i: u for u, i in nmap.items()} # reverse mapping of nodes
        S_tent = self.S | {int(w)}
        # h = nx.dag_longest_path_length(arb) # Can be optimized
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
                    prob_activated_earlier = np.product([1 - sum([AP_matrix[nmap[w], j]*self.G.get_edge_data(w, u)["weight"]
                                                                  for j in range(t-2)])
                                                         for w in arb.predecessors(u)])
                else:
                    prob_activated_earlier = 0
                prob_unactivated_by_now = np.product([1 - sum([AP_matrix[nmap[w], j]*self.G.get_edge_data(w, u)["weight"]
                                                               for j in range(t-1)])
                                                      for w in arb.predecessors(u)])
                AP_matrix[nmap[w], t] = prob_activated_earlier - prob_unactivated_by_now
        return sum([AP_matrix[nmap[v],t]*self.q**(t+1) for t in range(h)])

    def MIIA(self, v):
        arb = nx.DiGraph()
        arb.add_node(v)
        for u in self.G.nodes:
            new_nodes = self.pairwise_paths[int(u)][int(v)]
            ppp = self.pairwise_ppp[int(u)][int(v)]
            if ppp >= self.theta:
                nx.add_path(arb, new_nodes)
                # arb.add_weighted_edges_from([(new_nodes[i], new_nodes[i+1], self.G.get_edge_data(new_nodes[i], new_nodes[i+1])["weight"]) for i in range(len(new_nodes)-1)])
        h = nx.dag_longest_path_length(arb, weight=None)
        return arb, h
    
    def MIOA(self, v):
        arb = nx.DiGraph()
        for u in self.G.nodes:
            new_nodes = self.pairwise_paths[int(v)][int(u)]
            ppp = self.pairwise_ppp[int(v)][int(u)]
            #print(ppp)
            if ppp >= self.theta:
                nx.add_path(arb, new_nodes)
                # arb.add_weighted_edges_from([(new_nodes[i], new_nodes[i+1], self.G.get_edge_data(new_nodes[i], new_nodes[i+1])["weight"]) for i in range(len(new_nodes)-1)])
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

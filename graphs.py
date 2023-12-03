'''
Code to generate randomly* sampled subgraphs of the Epinions dataset
There's also code to generate synthetic networks.
All hyperparameters have been tuned to produce graphs with a similar average degree as the Epinions dataset

*Not uniformly sampled, but using random walk with teleportation as a heurstic
'''
import networkx as nx
import random
import numpy as np

def print_graph_info(G):
    print("Number of nodes:", len(G.nodes)) 
    print("Number of edges:", len(G.edges)) 
    print("Average in-degree:", sum(dict(G.in_degree).values()) / len(G.nodes))
    print("Average out-degree:", sum(dict(G.out_degree).values()) / len(G.nodes))

def random_walk_subgraph(G, target_num_nodes, directed=True, max_iters=1e6):
    # Choose a random starting node
    start_node = random.choice(list(G.nodes()))

    # Perform random walk
    current_node = start_node
    subgraph_nodes = set([current_node])

    iters = 0
    while len(subgraph_nodes) < target_num_nodes and iters < max_iters:
        neighbors = list(G.successors(current_node)) if directed else list(G.neighbors(current_node))
        if not neighbors:
            if len(subgraph_nodes) < 10:
                # Restart at randomly sampled node from the full graph
                current_node = random.choice(list(G.nodes()))
                subgraph_nodes = set([current_node])
            else:
                # Restart at another randomly sampled node within the current subgraph
                current_node = random.choice(list(subgraph_nodes))
        else:
            current_node = random.choice(neighbors)
            subgraph_nodes.add(current_node)

        iters += 1
        if iters == max_iters:
            print("Reached max iterations")

    # Create the subgraph
    subgraph = G.subgraph(subgraph_nodes)
    return subgraph

def get_graph(graph_name):
    n = 100
    if graph_name == "erdos_renyi":
        p = 0.3
        #G = nx.fast_gnp_random_graph(75877, 0.0002, seed=1, directed=True) # Random graph
        G = nx.fast_gnp_random_graph(n, p, seed=1, directed=True)
    elif graph_name == "barabasi_albert":
        G = nx.barabasi_albert_graph(75877, 7, seed=1, create_using=nx.DiGraph) # BA graph
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = np.random.uniform(0, 1)
    else:
        G = nx.read_edgelist('soc-Epinions1.txt.gz', create_using=nx.DiGraph)
        if graph_name == "epinions_subgraph":
            G = random_walk_subgraph(G, n, directed=True)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = np.random.uniform(0, 1)
    # Print graph information
    print_graph_info(G)
    return G
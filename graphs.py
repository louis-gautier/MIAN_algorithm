'''
Code to generate randomly* sampled subgraphs of the Epinions dataset
There's also code to generate synthetic networks.
All hyperparameters have been tuned to produce graphs with a similar average degree as the Epinions dataset

*Not uniformly sampled, but using random walk with teleportation as a heurstic
'''
import networkx as nx
import random

def print_graph_info(G, directed=True):
    print("Number of nodes:", len(G.nodes)) 
    print("Number of edges:", len(G.edges)) 
    if directed:
        print("Average in-degree:", sum(dict(G.in_degree).values()) / len(G.nodes))
        print("Average out-degree:", sum(dict(G.out_degree).values()) / len(G.nodes))
    else:
        print("Average degree:", sum(dict(G.degree).values()) / len(G.nodes))

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
              # Restart at another randomly sampled node within the current subgraph
              current_node = random.choice(list(subgraph_nodes))
            else:
              # Restart at randomly sampled node from the full graph
              current_node = random.choice(list(G.nodes()))
        else:
            current_node = random.choice(neighbors)
            subgraph_nodes.add(current_node)

        iters += 1

    # Create the subgraph
    subgraph = G.subgraph(subgraph_nodes)
    return subgraph

if __name__ == '__main__':
    
    directed = True # Set directed to True or False based on your graph
    create_as = nx.DiGraph if directed else nx.Graph
    n = 100 if directed else 250
    # G = nx.read_edgelist('soc-Epinions1.txt.gz', create_using=create_as) # Random subgraph of Epinions
    G = nx.fast_gnp_random_graph(75877, 0.0002, seed=1, directed=directed) # Random graph
    # G = nx.barabasi_albert_graph(75877, 7, seed=1) # BA graph

    # Generate the subgraph using random walk
    subgraph = random_walk_subgraph(G, n, directed=directed)  

    # Print graph information
    print_graph_info(subgraph, directed=directed)
import sys
from graphs import get_graph
from figures import plot_figures
import networkx as nx
from MIAN import MIAN

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py <graph> <algorithm> <k> <q>")
        sys.exit(1)

    graph_name = sys.argv[1]
    algorithm = sys.argv[2]
    k = int(sys.argv[3])
    q = float(sys.argv[4])
    assert(graph_name in ["erdos_renyi", "barabasi_albert", "epinions_subgraph", "epinions"])
    assert(algorithm in ["MIAN", "greedy"])

    graph = get_graph(graph_name)
    if algorithm == "MIAN":
        theta = 1/160 # from the paper
        alg = MIAN(graph, q, k, theta)
        optimal_seed_set = alg.run()
        plot_figures(optimal_seed_set)
    else:
        raise NotImplementedError
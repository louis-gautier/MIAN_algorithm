from matplotlib import pyplot as plt
from MC_simulation import ICN
import sys
from graphs import get_graph
from MIAN import MIAN
import numpy as np
import pandas as pd
from greedy import Greedy

def estimate_PIS(g, S, q, M=100):
    PIS_estimate = 0.0
    for i in range(M):
        PIS_estimate += ICN(g, S, q).positive_influence_spread()
    return PIS_estimate/M


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python figure3.py <graph>")
        sys.exit(1)

    graph_name = sys.argv[1]
    assert(graph_name in ["erdos_renyi", "barabasi_albert", "epinions_subgraph", "epinions"])

    # Constants
    theta = 1/160 # from the MIAN paper

    graph = get_graph(graph_name)
    qs = [0.7, 0.9]
    kmax = 50
    optimal_seeds_MIAN = {q: [] for q in qs}
    optimal_seeds_greedy = {q: [] for q in qs}

    for q in qs:
        #mian = MIAN(graph, q, kmax, theta, "results/"+graph_name+"_MIAN_k"+str(kmax)+"_q"+str(q)+".txt")
        #optimal_seeds_MIAN[q] = mian.run()
        greedy = Greedy(graph, kmax, lambda S: estimate_PIS(graph, S, q), "results/"+graph_name+"_greedy_k"+str(kmax)+"_q"+str(q)+".txt")
        optimal_seeds_greedy[q] = greedy.run()
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
        print(i)
        PIS_estimate += ICN(g, S, q).positive_influence_spread()
    return PIS_estimate/M


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python figure2_run.py <graph> <algorithm>")
        sys.exit(1)

    graph_name = sys.argv[1]
    algorithm = sys.argv[2]
    assert(graph_name in ["erdos_renyi", "barabasi_albert", "epinions_subgraph", "epinions", "wiki_vote", "wiki_vote_subgraph", "hept", "hept_subgraph"])
    assert(algorithm in ["MIAN", "greedy"])

    # Constants
    theta = 0.4

    graph = get_graph(graph_name)
    #qs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    qs = [0.9]
    ks = [1, 10, 20, 30, 40, 50]
    kmax = max(ks)
    optimal_seeds = {q: [] for q in qs}

    for q in qs:
        if algorithm == "MIAN":
            alg = MIAN(graph, q, kmax, theta, "results/"+graph_name+"_MIAN_k"+str(kmax)+"_q"+str(q)+".txt")
            optimal_seeds[q] = alg.run()
        else:
            alg = Greedy(kmax, lambda S: estimate_PIS(graph, S, q), "results/"+graph_name+"_greedy_k"+str(kmax)+"_q"+str(q)+".txt")
            optimal_seeds[q] = alg.run()
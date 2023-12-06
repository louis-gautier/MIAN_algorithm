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
    if len(sys.argv) != 3:
        print("Usage: python figure2.py <graph> <algorithm>")
        sys.exit(1)

    graph_name = sys.argv[1]
    algorithm = sys.argv[2]
    assert(graph_name in ["erdos_renyi", "barabasi_albert", "epinions_subgraph", "epinions"])
    assert(algorithm in ["MIAN", "greedy"])

    # Constants
    theta = 1/160 # from the MIAN paper
    run_algorithms = True

    graph = get_graph(graph_name)
    qs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ks = [1, 10, 20, 30, 40, 50]
    kmax = max(ks)
    optimal_seeds = {q: [] for q in qs}

    if run_algorithms:
        for q in qs:
            if algorithm == "MIAN":
                alg = MIAN(graph, q, kmax, theta, "results/"+graph_name+"_MIAN_k"+str(kmax)+"_q"+str(q)+".txt")
                optimal_seeds[q] = alg.run()
            else:
                alg = Greedy(kmax, lambda S: estimate_PIS(graph, S, q), "results/"+graph_name+"_greedy_k"+str(kmax)+"_q"+str(q)+".txt")
                optimal_seeds[q] = alg.run()

    PIS_df = pd.DataFrame(columns=['q', 'k', 'PIS'])
    
    for q in qs:
        for k in ks:
            optimal_S = optimal_seeds[q][:k]
            PIS_df.loc[len(PIS_df.index)] = [q, k, estimate_PIS(graph, optimal_S, q)]

    PIS_df.to_csv("results/PIS_"+algorithm+"_"+graph_name+".csv")

    PIS_df = PIS_df.groupby("k")
    for k in ks:
        plt.plot(qs, PIS_df[k]["PIS"], label=k)
    plt.xlabel("q")
    plt.ylabel("Positive Influence Spread")
    plt.savefig("figures/figure2_"+algorithm+"_"+graph_name+".csv")
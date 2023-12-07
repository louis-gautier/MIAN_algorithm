from matplotlib import pyplot as plt
from MC_simulation import ICN
import sys
from graphs import get_graph
import numpy as np
import pandas as pd

def estimate_PIS(g, S, q, M=100):
    PIS_estimate = 0.0
    for i in range(M):
        PIS_estimate += ICN(g, S, q).positive_influence_spread()
    return PIS_estimate/M


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python figure2_plot.py <graph> <algorithm>")
        sys.exit(1)

    graph_name = sys.argv[1]
    algorithm = sys.argv[2]
    assert(graph_name in ["erdos_renyi", "barabasi_albert", "epinions_subgraph", "epinions", "wiki_vote", "wiki_vote_subgraph", "hept", "hept_subgraph"])
    assert(algorithm in ["MIAN", "greedy"])


    graph = get_graph(graph_name)
    #qs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    qs = [0.9]
    ks = [1, 10, 20, 30, 40, 50]
    kmax = max(ks)

    PIS_df = pd.DataFrame(columns=['q', 'k', 'PIS'])
    
    for q in qs:
        print("q=", q)
        file_name = "results/"+graph_name+"_"+algorithm+"_k"+str(kmax)+"_q"+str(q)+".txt"
        with open(file_name, 'r') as file:
            optimal_seeds = file.read().splitlines()
        for k in ks:
            print("k=", k)
            optimal_S = set(optimal_seeds[:k])
            PIS_df.loc[len(PIS_df.index)] = [q, k, estimate_PIS(graph, optimal_S, q)]

    PIS_df.to_csv("results/PIS_"+algorithm+"_"+graph_name+".csv")

    PIS_df = PIS_df.groupby("k")
    for k in ks:
        plt.plot(qs, PIS_df.get_group(k)["PIS"], label=k)
    plt.xlabel("q")
    plt.ylabel("Positive Influence Spread")
    plt.savefig("figures/figure2_"+algorithm+"_"+graph_name+".png")
from matplotlib import pyplot as plt
from MC_simulation import ICN
import sys
from graphs import get_graph
from MIAN import MIAN
import numpy as np
import pandas as pd
from greedy import Greedy

def estimate_PIS(g, S, q, M=200):
    PIS_estimate = 0.0
    for i in range(M):
        PIS_estimate += ICN(g, S, q).positive_influence_spread()
    return PIS_estimate/M


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python figure3.py <graph>")
        sys.exit(1)

    graph_name = sys.argv[1]
    assert(graph_name in ["erdos_renyi", "barabasi_albert", "epinions_subgraph", "epinions", "hept_subgraph"])

    graph = get_graph(graph_name, n=500)
    qs = [0.9]
    #ks = list(range(0,51,5))
    ks = list(range(0,50))
    #kmax = max(ks)
    kmax = 50

    PIS_df = pd.DataFrame(columns=['alg', 'q', 'k', 'PIS'])
    
    for q in qs:
        MIAN_FILE = "results/"+graph_name+"_MIAN_k"+str(kmax)+"_q"+str(q)+".txt"
        with open(MIAN_FILE, 'r') as file:
            optimal_seeds_MIAN = file.read().splitlines()
        for k in ks:
            print("k=",k)
            optimal_S_MIAN = set(optimal_seeds_MIAN[:k])
            PIS_estimate = estimate_PIS(graph, optimal_S_MIAN, q)
            PIS_df.loc[len(PIS_df.index)] = ['MIAN', q, k, PIS_estimate]
            print(PIS_estimate)
            #optimal_S_greedy = optimal_seeds_greedy[q][:k]
            #PIS_df.loc[len(PIS_df.index)] = ['greedy', q, k, estimate_PIS(graph, optimal_S_greedy, q)]

    PIS_df.to_csv("results/PIS_figure3_"+graph_name+".csv")

    PIS_df_alg = PIS_df.groupby("alg")
    #for alg in ["MIAN", "greedy"]:
    for alg in ["MIAN"]:
        PIS_df_byq = PIS_df_alg.get_group(alg).groupby("q")
        for q in qs:
            plt.plot(ks, PIS_df_byq.get_group(q)["PIS"], label="q="+str(q)+", algorithm: "+alg)
    plt.xlabel("k")
    plt.ylabel("Positive Influence Spread")
    plt.savefig("figures/figure3_"+graph_name+".png")
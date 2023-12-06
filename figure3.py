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
    run_algorithms = True

    np.random.seed(42)
    graph = get_graph(graph_name)
    qs = [0.7, 0.9]
    ks = list(range(0,51,5))
    kmax = max(ks)
    optimal_seeds_MIAN = {qs: [] for q in qs}
    optimal_seeds_greedy = {qs: [] for q in qs}

    if run_algorithms:
        for q in qs:
            mian = MIAN(graph, q, kmax, theta)
            optimal_seeds_MIAN[q] = mian.run()
            greedy = Greedy(kmax, lambda S: estimate_PIS(graph, S, q), graph)
            optimal_seeds_greedy[q] = greedy.run()

    PIS_df = pd.DataFrame(columns=['alg', 'q', 'k', 'PIS'])
    
    for q in qs:
        for k in ks:
            optimal_S_MIAN = optimal_seeds_MIAN[q][:k]
            PIS_df.loc[len(PIS_df.index)] = ['MIAN', q, k, estimate_PIS(graph, optimal_S_MIAN, q)]
            optimal_S_greedy = optimal_seeds_greedy[q][:k]
            PIS_df.loc[len(PIS_df.index)] = ['greedy', q, k, estimate_PIS(graph, optimal_S_greedy, q)]

    PIS_df.to_csv("results/PIS_figure3_"+graph_name+".csv")

    PIS_df_alg = PIS_df.groupby("alg")
    for alg in ["MIAN", "greedy"]:
        PIS_df_byk = PIS_df[alg].groupby("k")
        for q in qs:
            plt.plot(qs, PIS_df_byk[k]["PIS"], label="q="+str(q)+", algorithm: "+alg)
    plt.xlabel("k")
    plt.ylabel("Positive Influence Spread")
    plt.savefig("figures/figure3_"+graph_name+".csv")
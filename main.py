#============================================================================================
# Name        : main.py
# Author      : Samuel Marchal, Sebastian Szyller
# Version     : 1.0
# Copyright   : Copyright (C) Secure Systems Group, Aalto University {https://ssg.aalto.fi/}
# License     : This code is released under Apache 2.0 license
#============================================================================================

from clustering import RecAgglo, SampleClust, AggloClust
import numpy as np
import pandas as pd
from parsing import Parser


def main():
    parser = Parser()
    args = parser.args

    infile = args.infile
    outfile = args.outfile
    verbose = args.verbose
    skip_index = args.skip_index
    delta_a = args.delta_a
    delta_fc = args.delta_fc
    d_max = args.d_max
    rho_mc = args.rho_mc
    rho_s = args.rho_s
    weights = list(map(float, args.weight.strip('[]').split(',')))
    algorithm = args.algo

    df = pd.read_csv(infile, dtype='str')

    if len(weights) != df.shape[1]:
        weights = np.ones(df.shape[1])

    if skip_index:
        weights[0] = 0. #index column weight set to 0 and not considered during clustering

    if verbose:
        print("ARGS:")
        print("\tinfile:", infile)
        print("\toutfile:", outfile)
        print("\tweight:", weights)
        print("\tdelta_a:", delta_a)
        print("\tdelta_fc:", delta_fc)
        print("\td_max:", d_max)
        print("\trho_mc:", rho_mc)
        print("\trho_s:", rho_s)

    if verbose:
        print("\ninput shape:", df.shape)


    if algorithm == 0:
        merged = np.append(df.values,np.zeros((df.shape[0],1)), axis=1)
        clusters = RecAgglo(merged, delta_a, delta_fc, d_max, rho_s, rho_mc, weights, verbose)

    else:
        to_cluster = df.values
        if algorithm == 1:
            clusters = SampleClust(to_cluster, rho_s, rho_mc, weights)
        elif algorithm == 2:
            clusters = AggloClust(to_cluster, 'single', d_max, weights)

    df_clusters = pd.DataFrame(clusters)
    df_clusters.to_csv(outfile)

if __name__ == "__main__":
    main()
